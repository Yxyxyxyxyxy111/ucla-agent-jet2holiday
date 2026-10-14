"""Local 7-seat Werewolf full-match simulation (no platform server).

From the repository root:
    python tests/test_full_werewolf.py --mode mock
    python tests/test_full_werewolf.py --mode llm --max-speeches 2

The LLM mode reuses agent.werewolf_agent.WerewolfAgent and the configured
provider, so it incurs API usage. Mock mode makes NO API calls.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLES = {0: 'villager', 1: 'villager', 2: 'wolf', 3: 'villager',
         4: 'villager', 5: 'wolf', 6: 'seer'}


@dataclass
class Action:
    action_id: str
    label: str


@dataclass
class Chat:
    index: int
    sender: int
    content: str
    recipients: list[int] | None = None


class LocalState:
    def __init__(self, raw: dict, phase: str, actions: list[Action], messages: list[Chat], version: int):
        self.raw = raw
        self.phase = phase
        self.legal_actions = actions
        self.new_messages = messages
        self.state_version = version
        self.observation = raw.get('observation')


def fresh_agents(mode: str):
    if mode == 'llm':
        from dotenv import load_dotenv
        from agent.werewolf_agent import WerewolfAgent
        from examples.llm.providers import provider_from_env
        load_dotenv()
        # Separate per-player agent memory; providers can share one transport/client.
        provider = provider_from_env()
        return {seat: WerewolfAgent(provider) for seat in ROLES}, provider.model
    return {}, 'mock (no API calls)'


class Match:
    def __init__(self, mode: str, max_speeches: int, max_days: int, seed: int, out: Path):
        self.mode = mode
        self.agents, self.model = fresh_agents(mode)
        self.max_speeches = max_speeches
        self.max_days = max_days
        self.rng = random.Random(seed)
        self.alive = set(ROLES)
        self.dead = []
        self.votes = []
        self.investigations: dict[int, list[dict]] = {6: []}
        self.no_elimination = 0
        self.day = 1
        self.messages: list[Chat] = []
        self.version = 0
        self.turn_count = 0
        self.records: list[dict] = []
        self.out = out

    def write(self, event: str, **data):
        item = {'event': event, 'day': self.day, **data}
        self.records.append(item)
        if event == 'speech':
            description = f"Player{data['seat']} ({ROLES[data['seat']]})：{data['text']}"
        elif event == 'vote':
            description = f"Player{data['seat']} -> " + ('弃票' if data['target'] == 7 else f"Player{data['target']}")
        else:
            description = data.get('text', '') if event in ('night', 'notice') else ''
        if description:
            print('  ' + description, flush=True)

    def context(self, seat: int):
        return SimpleNamespace(game_type='werewolf', seat_position=seat, session_id='local-fixed-roles')

    def game_record(self, phase: str):
        return {
            'day': self.day, 'phase': phase,
            'alive': [f'Player{x}' for x in sorted(self.alive)],
            'dead': list(self.dead), 'vote_history': list(self.votes),
            'no_elimination_streak': self.no_elimination,
        }

    def observation(self, seat: int):
        obs = {'role': ROLES[seat], 'your_position': seat}
        if ROLES[seat] == 'wolf':
            obs['wolf_teammate'] = next(x for x in (2, 5) if x != seat)
            obs['wolves'] = [2, 5]
        elif ROLES[seat] == 'seer':
            obs['investigations'] = list(self.investigations[6])
        return obs

    def state(self, seat: int, *, phase: str, engine_phase: str, actions: list[Action] | None = None):
        raw = {
            'your_position': seat,
            'observation': self.observation(seat),
            'game_state': self.game_record(engine_phase),
            'players': [{'position': i, 'name': f'Player{i}'} for i in ROLES],
            'messaging_seconds_left': 120,
        }
        self.version += 1
        # The real game exposes the day's discussion window. Each participant
        # sees the current public messages; own agent deduplicates by index.
        return LocalState(raw, phase, actions or [], list(self.messages) if phase == 'messaging' else [], self.version)

    def action(self, seat: int, kind: str, targets: list[int]):
        actions = [Action(str(t), ('Abstain' if t == 7 else f'{kind} Player{t}')) for t in targets]
        if self.mode == 'mock':
            # Fixed deterministic baseline: seer targets descending uninvestigated seats;
            # wolves hunt seer if possible; good players use revealed roles and day-1 tie-breaks.
            if kind == 'Investigate':
                prior = {r['target'] for r in self.investigations[6]}
                target = next((x for x in [2, 5, 1, 3, 4, 0] if x in targets and x not in prior), targets[0])
            elif kind == 'Kill':
                target = 6 if 6 in targets else (1 if 1 in targets else targets[0])
            else:
                if seat in (2, 5):
                    target = next((x for x in [1, 6, 0, 3, 4] if x in targets), targets[0])
                else:
                    # Good-side mock decisions use PUBLIC evidence only:
                    # a seer claimant's accusation and completed vote history.
                    public_reports = []
                    for m in self.messages:
                        if m.sender == 6 and 'seer' in m.content.lower() and 'wolf' in m.content.lower():
                            import re
                            match = re.search(r'Player([0-6]): wolf', m.content, re.I)
                            if match:
                                public_reports.append(int(match.group(1)))
                    suspects = [x for x in public_reports if x in targets]
                    if not suspects:
                        # Look for players who voted alongside a publicly revealed wolf.
                        confirmed_dead_wolves = {int(row['player'].replace('Player', ''))
                                                for row in self.dead if row['role'] == 'wolf'}
                        for history_entry in self.votes:
                            for vote in history_entry['votes']:
                                if vote['seat'] in confirmed_dead_wolves:
                                    suspects.extend(other['seat'] for other in history_entry['votes']
                                                    if other['target'] == vote['target']
                                                    and other['seat'] != vote['seat'] and other['seat'] in targets)
                    target = next((x for x in suspects if x in targets),
                                  next((x for x in [1, 0, 3, 4, 6, 2, 5] if x in targets), targets[0]))
            return target
        state = self.state(seat, phase='moving', engine_phase={'Kill': 'night_wolf', 'Investigate': 'night_seer', 'Vote to eliminate': 'day_vote'}[kind], actions=actions)
        result = self.agents[seat].choose_action(state, self.context(seat))
        # Works with either LegalAction or WithReasoning, and rejects illegal decisions.
        chosen = getattr(result, 'action', result)
        if hasattr(chosen, 'action_id'):
            identifier = chosen.action_id
        else:
            identifier = str(chosen)
        if identifier not in {a.action_id for a in actions}:
            raise RuntimeError(f'Player{seat} produced illegal {kind} action: {identifier}')
        return int(identifier)

    def mock_speech(self, seat: int, round_idx: int):
        role = ROLES[seat]
        if round_idx == 0:
            if role == 'seer':
                reports = self.investigations[6]
                wolf = next((r for r in reports if r['result'] == 'wolf' and r['target'] in self.alive), None)
                if wolf:
                    return f"I am the seer. Night {wolf['night']} I checked Player{wolf['target']}: wolf. Vote Player{wolf['target']}."
                return 'I want everyone to explain their suspicions before we choose a target.'
            if role == 'wolf':
                return ('We need evidence, not guesses. Who has a concrete reason for their vote?'
                        if seat == 2 else 'I want to hear the proposed votes and the reasons behind them.')
            return {
                0: 'Please give one suspect and a concrete reason. We should avoid splitting votes.',
                1: 'I want claims and voting reasons recorded before we eliminate anyone.',
                3: 'I will compare seer claims with the night results and past votes.',
                4: 'No unsupported certainty. Let us agree on one candidate if possible.',
            }[seat]
        if role == 'wolf':
            exposed = any(m.sender == 6 and f'Player{seat}: wolf' in m.content for m in self.messages) and 6 in self.alive
            if exposed:
                other = next((p for p in sorted(self.alive) if p not in (seat, 6, 2, 5)), None)
                return (f"I'm the real seer. Night 1 I checked Player{other}: innocent. Player6 is lying. Vote Player6."
                        if other is not None else 'Player6 is lying about my role. Look at the voting record.')
            return 'My current vote would be based on the strongest public evidence, not a personal grudge.'
        if role == 'seer':
            return 'Please compare my exact investigation result with the public record and coordinate the vote.'
        return 'I am weighing the seer claims, the answers to our questions, and the existing vote record.'

    def discussion(self):
        print(f'\n=== Day {self.day}: discussion ===', flush=True)
        self.messages = []
        active = sorted(self.alive)
        done = set()
        for round_idx in range(self.max_speeches):
            for seat in active:
                if seat in done:
                    continue
                if self.mode == 'mock':
                    text = self.mock_speech(seat, round_idx)
                else:
                    from altruagent import TERMINATE_MESSAGING
                    state = self.state(seat, phase='messaging', engine_phase='day_vote')
                    result = self.agents[seat].choose_message(state, self.context(seat))
                    if result is TERMINATE_MESSAGING:
                        done.add(seat)
                        continue
                    text = result.content
                if not text:
                    done.add(seat)
                    continue
                self.messages.append(Chat(len(self.messages), seat, text))
                self.write('speech', seat=seat, text=text, round=round_idx+1)
            if len(done) == len(active):
                break

    def investigate(self):
        if 6 not in self.alive:
            return
        choices = sorted(self.alive - {6})
        target = self.action(6, 'Investigate', choices)
        row = {'night': self.day, 'target': target, 'result': 'wolf' if ROLES[target] == 'wolf' else 'villager'}
        self.investigations[6].append(row)
        self.write('investigation', seat=6, **row)
        print(f"  [private to Player6] Night {self.day}: Player{target} = {row['result']}", flush=True)

    def kill(self):
        wolf_targets = []
        for wolf in sorted(self.alive & {2, 5}):
            legal = sorted(self.alive - {2, 5})
            if legal:
                wolf_targets.append((wolf, self.action(wolf, 'Kill', legal)))
        if wolf_targets:
            count = {}
            for _, target in wolf_targets:
                count[target] = count.get(target, 0) + 1
            maximum = max(count.values())
            target = min(t for t, votes in count.items() if votes == maximum)
            self.eliminate(target, 'night_kill')
            self.write('night', text=f"狼人夜间选择 {wolf_targets}；Player{target} 被击杀，身份 {ROLES[target]}")

    def eliminate(self, target: int, cause: str):
        self.alive.remove(target)
        self.dead.append({'player': f'Player{target}', 'role': ROLES[target], 'cause': cause, 'day': self.day})

    def voting(self):
        print(f'\n=== Day {self.day}: voting (hidden until complete) ===', flush=True)
        tally: dict[int, int] = {}
        all_votes = []
        for seat in sorted(self.alive):
            legal = sorted(self.alive - {seat}) + [7]
            target = self.action(seat, 'Vote to eliminate', legal)
            all_votes.append({'seat': seat, 'target': target})
            if target != 7:
                tally[target] = tally.get(target, 0) + 1
        for row in all_votes:
            self.write('vote', **row)
        if not tally:
            winner = None
        else:
            highest = max(tally.values())
            leaders = [seat for seat, n in tally.items() if n == highest]
            winner = leaders[0] if len(leaders) == 1 else None
        self.votes.append({'day': self.day, 'votes': all_votes, 'tally': tally, 'eliminated': winner})
        if winner is None:
            self.no_elimination += 1
            self.write('notice', text=f'平票或全部弃票，无人出局；累计连续 {self.no_elimination} 天')
        else:
            self.no_elimination = 0
            self.eliminate(winner, 'vote')
            self.write('notice', text=f'Player{winner} 被投出，真实身份：{ROLES[winner]}')

    def winner(self):
        wolves = len(self.alive & {2, 5})
        good = len(self.alive) - wolves
        if wolves == 0:
            return 'villagers'
        if wolves >= good or self.no_elimination >= 3:
            return 'wolves'
        return None

    def run(self):
        self.write('notice', text=f'固定身份：村民 0/1/3/4；预言家 6；狼人 2/5；模型 {self.model}')
        try:
            for day in range(1, self.max_days + 1):
                self.day = day
                print(f'\n=== Night {day} ===', flush=True)
                if day == 1:
                    self.write('night', text='第一夜和平夜，狼人不能杀人')
                # Both night actions resolve at dawn. The seer still gets an
                # investigation even if wolves kill the seer that night.
                self.investigate()
                if day >= 2:
                    self.kill()
                    if self.winner():
                        break
                self.discussion()
                self.voting()
                if self.winner():
                    break
            winner = self.winner() or 'unfinished (max-days limit)'
            self.write('result', winner=winner, alive=sorted(self.alive))
            print(f'\n=== RESULT: {winner} ===', flush=True)
            return winner
        finally:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.out.write_text(json.dumps({
                'roles': ROLES, 'mode': self.mode, 'model': self.model,
                'records': self.records, 'vote_history': self.votes,
                'investigations': self.investigations, 'winner': self.winner(),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'完整对局 JSON 日志：{self.out}', flush=True)


def main():
    p = argparse.ArgumentParser(description='Fixed-roles local 7-person Werewolf simulation')
    p.add_argument('--mode', choices=['mock', 'llm'], default='mock', help='llm actually calls configured API')
    p.add_argument('--max-speeches', type=int, default=2, help='speeches per seat per day (1..5)')
    p.add_argument('--max-days', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, default=Path('tests/last_full_werewolf.json'))
    args = p.parse_args()
    if not 1 <= args.max_speeches <= 5:
        p.error('--max-speeches must be between 1 and 5')
    if args.max_days < 1:
        p.error('--max-days must be positive')
    Match(args.mode, args.max_speeches, args.max_days, args.seed, args.out).run()


if __name__ == '__main__':
    main()
