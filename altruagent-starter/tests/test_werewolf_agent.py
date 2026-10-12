import json
from altruagent import DecisionContext, GameState, TERMINATE_MESSAGING
from agent.werewolf_agent import WerewolfAgent
from examples.llm.providers import ProviderError

CTX = DecisionContext('s', None, 'werewolf', 'a', 0)

class Provider:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.prompts = []
    def complete_structured(self, messages, name, schema, **kwargs):
        self.prompts.append(json.loads(messages[1]['content']))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer


def state(day=1, version=1, messages=None, seconds=100):
    return GameState.from_mcp_state({'session_id': 's', 'state_version': version,
        'phase': 'messaging', 'game_state': {'day': day, 'phase': 'day_vote'},
        'observation': {'role': 'seer'}, 'messaging_seconds_left': seconds,
        'new_messages': messages or [],
        'legal_actions': {'actions': [{'action_id': '7', 'label': 'Abstain'},
                                     {'action_id': '2', 'label': 'Vote Player2'}]}})


def test_illegal_action_retries_and_uses_exact_server_object():
    p = Provider({'action_id': '99', 'private_notes': 'bad'},
                 {'action_id': '2', 'private_notes': 'confirmed wolf'})
    a = WerewolfAgent(p)
    s = state()
    assert a.choose_action(s, CTX) is s.legal_actions[1]
    assert a.notes == 'confirmed wolf'
    assert len(p.prompts) == 2


def test_provider_failure_falls_back_without_abstaining():
    a = WerewolfAgent(Provider(ProviderError('failed'), ProviderError('failed')))
    assert a.choose_action(state(), CTX).action_id == '2'


def test_discussion_budget_uses_day_not_version_and_resets_next_day():
    p = Provider(*[{'decision': 'send', 'message': 'Please explain your vote.',
                   'private_notes': 'suspect 2'} for _ in range(4)])
    a = WerewolfAgent(p)
    for version in range(3):
        assert a.choose_message(state(version=version), CTX).content
    assert a.choose_message(state(version=99), CTX) is TERMINATE_MESSAGING
    assert a.choose_message(state(day=2), CTX).content


def test_history_survives_vote_and_private_notes_are_not_broadcast():
    p = Provider({'decision': 'send', 'message': 'Vote Player2.', 'private_notes': 'secret'},
                 {'action_id': '2', 'private_notes': 'secret'})
    a = WerewolfAgent(p)
    s = state(messages=[{'index': 1, 'sender': 2, 'content': 'I am the seer.'}])
    assert a.choose_message(s, CTX).content == 'Vote Player2.'
    a.choose_action(state(), CTX)
    assert p.prompts[1]['recent_discussion'][0]['text'] == 'I am the seer.'
    assert p.prompts[1]['private_notes'] == 'secret'


def test_low_clock_does_not_call_provider():
    p = Provider()
    assert WerewolfAgent(p).choose_message(state(seconds=3), CTX) is TERMINATE_MESSAGING
    assert not p.prompts


def test_invalid_speech_ends_round_and_repeated_messages_are_deduplicated():
    p = Provider(*[{'decision': 'send', 'message': 'x' * 241, 'private_notes': ''}] * 2)
    a = WerewolfAgent(p)
    s = state(messages=[{'index': 0, 'sender': 2, 'content': 'Hello'}])
    assert a.choose_message(s, CTX) is TERMINATE_MESSAGING
    assert a.choose_message(s, CTX) is TERMINATE_MESSAGING
    assert len(a.history) == 1
