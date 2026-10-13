import json
from altruagent import GameState, DecisionContext, TERMINATE_MESSAGING
from agent.werewolf_agent import WerewolfAgent

CTX = DecisionContext('s', None, 'werewolf', 'a', 0)
class Provider:
    def __init__(self, answers):
        self.answers=iter(answers)
        self.payloads=[]
    def complete_structured(self, messages, *args, **kwargs):
        self.payloads.append(json.loads(messages[1]['content']))
        return next(self.answers)

def state(role='wolf', teammates=None, alive=None, phase='messaging', messages=None):
    return GameState.from_mcp_state({'phase':phase, 'state_version':1,
        'observation':{'role':role, 'wolf_teammates': teammates or []},
        'players':[{'position':i,'name':f'P{i}'} for i in range(7)],
        'game_state':{'day':1,'phase':'night_wolf' if phase=='moving' else 'day_vote',
                      'alive': alive if alive is not None else [f'P{i}' for i in range(7)]},
        'new_messages':messages or [],
        'legal_actions':{'actions':[{'action_id':str(i)} for i in [1,2,3,4,6]]}})

def answer(recipient):
    return {'decision':'send','message':'Player2 looks like the Seer. Agree?',
            'recipient':recipient,'private_notes':'proposed Player2, no agreement yet'}

def test_send_to_confirmed_teammate_and_remember_reply_for_night():
    p=Provider([answer(5), {'action_id':'2','private_notes':'agreed target 2'}])
    a=WerewolfAgent(p)
    msg=a.choose_message(state(teammates=[5]),CTX)
    assert msg.recipients==[5]
    night=state(phase='moving',messages=[{'sender':5,'recipients':[0], 'content':'Agree, target Player2.', 'index':1}])
    assert a.choose_action(night,CTX).action_id=='2'
    assert p.payloads[1]['recent_discussion'][0]['recipients']==[0]

def test_villager_cannot_send_private_wolf_coordination():
    a=WerewolfAgent(Provider([answer(5),answer(5)]))
    assert a.choose_message(state(role='villager',teammates=[5]),CTX) is TERMINATE_MESSAGING

def test_unknown_dead_and_boolean_recipients_rejected():
    for recipient, teammates, alive in [(4,[5],None),(5,[5],['P0','P2']), (True,[1],None)]:
        a=WerewolfAgent(Provider([answer(recipient),answer(recipient)]))
        assert a.choose_message(state(teammates=teammates,alive=alive),CTX) is TERMINATE_MESSAGING

def test_public_send_and_server_target_complement():
    a=WerewolfAgent(Provider([answer(None)]))
    assert not a.choose_message(state(),CTX).recipients
    assert a._private_recipients(state(phase='moving'),CTX)==[5]
    assert a._private_recipients(state(),CTX)==[5]
