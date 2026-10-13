import sys,types,unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.scenario_test_agent import ScenarioRuleAgent,ScenarioTestAgent
fake=types.ModuleType('altruagent');fake.SendMessage=lambda text:N(content=text);fake.TERMINATE_MESSAGING=object()
def state(role,results=None,messages=None):
 return N(raw={'observation':{'role':role,'investigations':results or []},'game_state':{'day':1,'phase':'day_vote'}},observation='',state_version=1,new_messages=messages or [],legal_actions=[N(action_id=str(i)) for i in [0,1,3,4,5,6,7]])
CTX=N(session_id='fixture',seat_position=2,game_type='werewolf')
class ScenarioTests(unittest.TestCase):
 def test_seer_reports_only_actual_results_and_votes_confirmed_wolf(self):
  a=ScenarioRuleAgent();s=state('seer',[{'target':0,'result':'wolf'},{'target':1,'result':'villager'}])
  with patch.dict(sys.modules,{'altruagent':fake}):
   self.assertIn('Player0 is a wolf',a.choose_message(s,CTX).content)
  self.assertEqual(a.choose_action(s,CTX).action_id,'0')
 def test_no_result_never_fabricates_report(self):
  a=ScenarioRuleAgent()
  with patch.dict(sys.modules,{'altruagent':fake}):
   for _ in range(3):self.assertNotIn('My investigation found',a.choose_message(state('seer'),CTX).content)
   self.assertIs(a.choose_message(state('seer'),CTX),fake.TERMINATE_MESSAGING)
 def test_villager_reacts_to_claim_and_wolf_defends(self):
  m=N(index=1,sender=2,recipients=[],type='chat',content='I am the Seer. Player0 is a wolf.')
  with patch.dict(sys.modules,{'altruagent':fake}):
   a=ScenarioRuleAgent();ctx=N(session_id='s',seat_position=0)
   self.assertIn('dispute',a.choose_message(state('wolf',messages=[m]),ctx).content)
   a=ScenarioRuleAgent();a.choose_message(state('villager',messages=[m]),CTX)
   self.assertIn('Player0',a.choose_message(state('villager',messages=[m]),CTX).content)
 def test_rule_slots_never_create_llm(self):
  a=ScenarioTestAgent({0},llm_factory=lambda:self.fail('Unexpected model call'))
  self.assertIsInstance(a._agent(state('seer'),CTX),ScenarioRuleAgent)
if __name__=='__main__':unittest.main()
