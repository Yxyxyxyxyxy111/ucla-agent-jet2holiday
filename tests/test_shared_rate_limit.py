import runpy
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

MODULE=Path(__file__).resolve().parents[1]/'examples/llm/rate_limit.py'
S=runpy.run_path(str(MODULE))

class SharedLimitTests(unittest.TestCase):
    def test_seven_processes_share_three_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=str(Path(tmp)/'quota.db')
            S['SharedLimiter']('test',3,1000,20,path=db)
            script=f"""import runpy,time
s=runpy.run_path({str(MODULE)!r})
l=s['SharedLimiter']('test',3,1000,20,path={db!r})
try:
 l.acquire(100,time.monotonic()+.5)
 print('ok')
except s['RateLimitDeadline']:
 print('limited')
"""
            processes=[subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,text=True) for _ in range(7)]
            results=[p.communicate(timeout=5)[0].strip() for p in processes]
            self.assertEqual(results.count('ok'),3)
            self.assertEqual(results.count('limited'),4)
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM calls').fetchone()[0],3)

    def test_token_daily_cooldown_and_deadline_limits(self):
        for rpm,tpm,rpd,initial,following,cooldown in [(20,150,20,100,100,False),
                   (20,1000,1,100,100,False),(20,1000,20,0,100,True)]:
            with tempfile.TemporaryDirectory() as tmp:
                limiter=S['SharedLimiter']('test',rpm,tpm,rpd,path=Path(tmp)/'q.db')
                if initial:
                    limiter.acquire(initial,time.monotonic()+1)
                if cooldown:
                    limiter.backoff(60)
                start=time.monotonic()
                with self.assertRaises(S['RateLimitDeadline']):
                    limiter.acquire(following,start+.1)
                self.assertLess(time.monotonic()-start,.5)

    def test_oversized_request_does_not_consume_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            limiter=S['SharedLimiter']('test',3,100,20,path=Path(tmp)/'q.db')
            with self.assertRaises(S['RateLimitDeadline']):
                limiter.acquire(101,time.monotonic()+1)
            limiter.acquire(100,time.monotonic()+1)

if __name__=='__main__':
    unittest.main()
