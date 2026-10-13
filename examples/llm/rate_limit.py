"""Cross-process rolling-window limiter. SQLite stores counters, never prompts/keys."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time


class RateLimitDeadline(RuntimeError):
    pass


class SharedLimiter:
    def __init__(self, bucket, rpm, tpm, rpd, *, path=None):
        if min(rpm, tpm, rpd) <= 0:
            raise ValueError('Rate limits must be positive')
        self.bucket, self.rpm, self.tpm, self.rpd = bucket, rpm, tpm, rpd
        root = Path(tempfile.gettempdir()) / f'altruagent-rate-limit-{os.getuid()}'
        root.mkdir(mode=0o700, exist_ok=True)
        self.path = str(path or root / 'quota.sqlite3')
        with sqlite3.connect(self.path, timeout=1) as db:
            db.execute('CREATE TABLE IF NOT EXISTS calls (bucket TEXT, stamp REAL, tokens INTEGER)')
            db.execute('CREATE INDEX IF NOT EXISTS calls_bucket_time ON calls(bucket, stamp)')
            db.execute('CREATE TABLE IF NOT EXISTS cooldown (bucket TEXT PRIMARY KEY, until REAL)')
        os.chmod(self.path, 0o600)

    def acquire(self, tokens, deadline):
        if tokens > self.tpm:
            raise RateLimitDeadline('Request token estimate exceeds shared per-minute budget')
        while True:
            if time.monotonic() >= deadline:
                raise RateLimitDeadline('Shared quota wait exceeded decision deadline')
            try:
                with sqlite3.connect(self.path, timeout=min(0.2, max(.01, deadline-time.monotonic()))) as db:
                    db.execute('BEGIN IMMEDIATE')
                    now = time.time()
                    db.execute('DELETE FROM calls WHERE stamp <= ?', (now-86400,))
                    rows = db.execute('SELECT stamp,tokens FROM calls WHERE bucket=? ORDER BY stamp',
                                      (self.bucket,)).fetchall()
                    minute = [(stamp, n) for stamp,n in rows if stamp > now-60]
                    cooldown = db.execute('SELECT until FROM cooldown WHERE bucket=?', (self.bucket,)).fetchone()
                    wait = max(0, (cooldown[0] if cooldown else 0)-now)
                    if len(rows) >= self.rpd:
                        wait=max(wait, rows[0][0]+86400-now)
                    if len(minute) >= self.rpm:
                        wait=max(wait, minute[0][0]+60-now)
                    total=sum(n for _,n in minute)
                    for stamp,n in minute:
                        if total+tokens <= self.tpm:
                            break
                        wait=max(wait, stamp+60-now)
                        total-=n
                    if wait <= 0:
                        db.execute('INSERT INTO calls VALUES (?,?,?)', (self.bucket,now,tokens))
                        return
            except sqlite3.OperationalError:
                wait=.1
            remaining=deadline-time.monotonic()
            if wait >= remaining:
                raise RateLimitDeadline('Shared quota unavailable within decision deadline')
            time.sleep(min(max(wait,.02),.25,remaining))

    def backoff(self, seconds):
        with sqlite3.connect(self.path, timeout=1) as db:
            db.execute('INSERT INTO cooldown VALUES (?,?) ON CONFLICT(bucket) DO UPDATE '
                       'SET until=MAX(until,excluded.until)',(self.bucket,time.time()+seconds))


class LimitedProvider:
    def __init__(self, provider, limiter):
        self.provider, self.limiter, self.model = provider, limiter, provider.model

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        from .providers import ProviderError
        deadline=time.monotonic()+(timeout if timeout is not None else 20)
        # Approximation, not a tokenizer: English-heavy UTF-8 bytes / 3 plus
        # an estimated output allowance. Server limits remain authoritative.
        estimate=math.ceil(len(json.dumps([messages,schema],ensure_ascii=False).encode())/3)+int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512"))
        try:
            self.limiter.acquire(estimate,deadline)
        except RateLimitDeadline as exc:
            raise ProviderError(str(exc)) from None
        remaining=deadline-time.monotonic()
        if remaining <= 0:
            raise ProviderError('Shared quota exhausted decision deadline')
        try:
            return self.provider.complete_structured(messages,schema_name,schema,timeout=remaining)
        except ProviderError as exc:
            if 'HTTP 429' in str(exc):
                # All workers observe this pause; no immediate retry storm.
                self.limiter.backoff(60)
            raise


def wrap_from_env(provider, base_url):
    groq='api.groq.com' in base_url
    gemini='generativelanguage.googleapis.com' in base_url
    defaults=(26,7200,900) if groq else (4,225000,18) if gemini else None
    if defaults is None and not os.getenv('LLM_RATE_RPM'):
        return provider
    defaults=defaults or (20,10000,1000)
    limits=[int(os.getenv(name,str(default))) for name,default in
            zip(('LLM_RATE_RPM','LLM_RATE_TPM','LLM_RATE_RPD'),defaults)]
    # All keys/models on the same service share a conservative local budget.
    # Optional bucket override allows matching the actual service quota group.
    identity=os.getenv('LLM_RATE_BUCKET') or base_url.rstrip('/')
    bucket=hashlib.sha256(identity.encode()).hexdigest()
    return LimitedProvider(provider,SharedLimiter(bucket,*limits))
