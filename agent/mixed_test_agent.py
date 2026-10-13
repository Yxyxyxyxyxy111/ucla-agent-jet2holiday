"""Testing only: selected seats use LLM; remaining seats use inexpensive rules."""
from __future__ import annotations
import os
from .strategies import extract_role


def parse_seats(value):
    try:
        seats = {int(x.strip()) for x in value.split(',') if x.strip()}
    except ValueError:
        raise ValueError('TEST_LLM_SEATS must contain comma-separated seats 0..6') from None
    if not seats or any(x < 0 or x > 6 for x in seats):
        raise ValueError('TEST_LLM_SEATS must contain seats 0..6')
    return seats


class RuleAgent:
    def __init__(self):
        self.spoken_day = None
        self.investigated = set()

    def choose_action(self, state, context):
        if not state.legal_actions:
            raise ValueError('No legal actions supplied')
        game = state.raw.get('game_state') or {}
        phase = game.get('phase')
        options = sorted(state.legal_actions, key=lambda a: a.action_id)
        candidates = [a for a in options if a.action_id != '7'] or options
        if phase == 'night_seer':
            candidates = [a for a in candidates if a.action_id not in self.investigated] or candidates
            target = candidates[0]
            self.investigated.add(target.action_id)
        elif phase == 'night_wolf':
            # Both rule wolves use the same lowest server-legal target.
            target = candidates[0]
        else:
            # Deliberately simple baseline. No access to other seats' identities.
            target = candidates[0]
        print(f'[RULE Player{context.seat_position}] {phase}: action={target.action_id}', flush=True)
        return target

    def choose_message(self, state, context):
        from altruagent import SendMessage, TERMINATE_MESSAGING
        game = state.raw.get('game_state') or {}
        day = ('day',game['day']) if 'day' in game else ('version',state.state_version)
        if self.spoken_day == day:
            return TERMINATE_MESSAGING
        self.spoken_day = day
        public = [m for m in state.new_messages if not m.recipients and getattr(m,'type','chat')=='chat']
        if not public:
            text = "Let's hear from everyone before voting. Who do you suspect, and why?"
        else:
            candidates = []
            names = {p.get('name'):p.get('position') for p in state.raw.get('players') or [] if isinstance(p,dict)}
            for player in game.get('alive') or []:
                seat = player if type(player) is int else names.get(player)
                if type(seat) is int and seat != context.seat_position:
                    candidates.append(seat)
            text = (f'I suggest discussing Player{min(candidates)} before voting. '
                    'Please share evidence; this is not an investigation result.') if candidates else (
                    'Please explain your accusations using public evidence. I have no verified accusation to announce.')
        print(f'[RULE Player{context.seat_position}] public message',flush=True)
        return SendMessage(text)


class MixedTestAgent:
    def __init__(self, seats, *, llm_factory=None):
        self.seats = seats
        self.llm_factory = llm_factory
        self.delegate = None
        self.seat = None

    def _agent(self, state, context):
        if context.game_type != 'werewolf':
            raise ValueError('Mixed test mode supports werewolf only')
        if type(context.seat_position) is not int or not 0 <= context.seat_position <= 6:
            raise ValueError('Mixed test mode requires a verified seat 0..6')
        if self.seat is not None and self.seat != context.seat_position:
            raise ValueError('One agent instance cannot serve different seats')
        if self.delegate is None:
            self.seat = context.seat_position
            if self.seat in self.seats:
                if self.llm_factory is None:
                    from .werewolf_agent import create_agent
                    self.llm_factory = create_agent
                self.delegate = self.llm_factory()
                kind = 'LLM'
            else:
                self.delegate = RuleAgent()
                kind = 'RULE (no model calls)'
            print(f'[MIXED TEST Player{self.seat}] {kind}',flush=True)
        return self.delegate

    def choose_action(self, state, context):
        return self._agent(state,context).choose_action(state,context)

    def choose_message(self, state, context):
        return self._agent(state,context).choose_message(state,context)


def create_agent():
    from dotenv import load_dotenv
    load_dotenv()
    return MixedTestAgent(parse_seats(os.getenv('TEST_LLM_SEATS','0')))
