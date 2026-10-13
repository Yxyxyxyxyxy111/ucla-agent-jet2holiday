"""Standard-library tests for prompt routing; no API calls or SDK imports."""
import runpy
import unittest
from pathlib import Path

S = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'agent' / 'strategies.py'))

class GoodStrategyTests(unittest.TestCase):
    def test_seer_first_night_has_neighbor_rule_and_no_speech_layer(self):
        p=S['get_strategy']('seer',{'day':1,'phase':'night_seer'})
        self.assertIn(S['SEER_NIGHT_STRATEGY'],p)
        self.assertIn('modulo 7',p)
        self.assertNotIn(S['SEER_SPEAK_STRATEGY'],p)

    def test_opening_is_only_day_one_discussion(self):
        for role in ('seer','villager'):
            key=role.upper()+'_OPENING_STRATEGY'
            self.assertIn(S[key],S['get_strategy'](role,{'day':1},phase='messaging'))
            self.assertNotIn(S[key],S['get_strategy'](role,{'day':2},phase='messaging'))

    def test_defense_available_on_later_days_and_wolf_layers_absent(self):
        for role in ('seer','villager'):
            p=S['get_strategy'](role,{'day':3},phase='messaging')
            self.assertIn(S[role.upper()+'_DEFENSE_STRATEGY'],p)
            self.assertIn(S[role.upper()+'_ENDGAME_STRATEGY'],p)
            self.assertNotIn(S['WOLF_COUNTERCLAIM_STRATEGY'],p)
        self.assertIn(S['RISK_ANALYZE_STRATEGY'],S['get_strategy']('seer',{},phase='messaging'))

    def test_voting_routes_to_role_specific_constraints(self):
        for role in ('seer','villager'):
            p=S['get_strategy'](role,{},phase='day_vote')
            self.assertIn(S[role.upper()+'_VOTE_STRATEGY'],p)
            self.assertNotIn(S[role.upper()+'_SPEAK_STRATEGY'],p)
        self.assertIn('Never vote for a living player',S['get_strategy']('seer',{},phase='day_vote'))

    def test_wolf_routing_and_unknown_role(self):
        p=S['get_strategy']('wolf',{'day':1},phase='messaging')
        self.assertIn(S['WOLF_OPENING_STRATEGY'],p)
        self.assertIn(S['WOLF_COUNTERCLAIM_STRATEGY'],p)
        self.assertIn(S['UNKNOWN_ROLE_STRATEGY'],S['get_strategy'](None))

if __name__ == '__main__':
    unittest.main()
