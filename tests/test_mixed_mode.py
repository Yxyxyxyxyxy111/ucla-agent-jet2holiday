import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace as N
# Minimal source extraction keeps routing tests independent of third-party SDKs.
import ast
p=Path(__file__).resolve().parents[1]/'agent/mixed_test_agent.py'
t=ast.parse(p.read_text());t.body=[n for n in t.body if not isinstance(n,ast.ImportFrom)]
s={};exec(compile(t,str(p),'exec'),s)

class MixedTests(unittest.TestCase):
    def test_only_selected_seat_creates_model(self):
        calls=[]
        def factory():
            calls.append(1)
            return N(choose_action=lambda *_:'model')
        state=N(raw={'game_state':{'phase':'day_vote'}},legal_actions=[N(action_id='7'),N(action_id='2')])
        for seat in range(7):
            agent=s['MixedTestAgent']({0},llm_factory=factory)
            ctx=N(game_type='werewolf',seat_position=seat)
            result=agent.choose_action(state,ctx)
            self.assertEqual(result if seat==0 else result.action_id,'model' if seat==0 else '2')
        self.assertEqual(len(calls),1)

    def test_seat_validation(self):
        self.assertEqual(s['parse_seats']('0,5'),{0,5})
        for value in ('','7','-1','wolf'):
            with self.assertRaises(ValueError):s['parse_seats'](value)
        with self.assertRaises(ValueError):
            s['MixedTestAgent']({0})._agent(None,N(game_type='werewolf',seat_position=None))

    def test_night_targets_are_server_options(self):
        a=s['RuleAgent']()
        opts=[N(action_id='4'),N(action_id='2')]
        state=N(raw={'game_state':{'phase':'night_seer'}},legal_actions=opts)
        ctx=N(seat_position=1)
        self.assertEqual(a.choose_action(state,ctx).action_id,'2')
        self.assertEqual(a.choose_action(state,ctx).action_id,'4')
        state.raw['game_state']['phase']='night_wolf'
        self.assertIn(a.choose_action(state,ctx),opts)

if __name__=='__main__':unittest.main()
