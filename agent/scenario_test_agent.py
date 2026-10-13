"""Optional mixed mode with role-aware scripted opponents; no extra LLM calls."""
from __future__ import annotations
import hashlib
import os
import re
from .mixed_test_agent import MixedTestAgent, RuleAgent, parse_seats
from .strategies import extract_role


class ScenarioRuleAgent(RuleAgent):
    def __init__(self):
        super().__init__()
        self.round = None
        self.messages = 0
        self.history = []
        self.seen = set()
        self.results = {}

    def _update(self, state):
        day = (state.raw.get('game_state') or {}).get('day')
        for m in state.new_messages:
            key=(day,m.index,m.sender,m.content)
            if key not in self.seen and getattr(m,'type','chat')=='chat' and not m.recipients:
                self.seen.add(key)
                self.history.append((day,m.sender,m.content))
        obs=state.raw.get('observation',state.observation)
        # Only explicit server results are accepted, never public claims.
        if extract_role(obs)=='seer' and isinstance(obs,dict):
            for item in obs.get('investigations',[]) if isinstance(obs.get('investigations',[]),list) else []:
                if not isinstance(item,dict):continue
                target=item.get('target');result=item.get('result')
                if type(target)is int and 0<=target<=6 and result in ('wolf','villager'):
                    self.results[target]=result
        return extract_role(obs)

    def _claims(self, day):
        claims=[]
        for d,sender,text in self.history:
            if d!=day:continue
            if re.search(r'\b(?:seer|investigat\w*)\b',text,re.I):
                for target in re.findall(r'Player([0-6])\s+(?:is|was)\s+(?:a\s+)?wolf\b',text,re.I):
                    claims.append((sender,int(target)))
        return claims

    def _pick(self, options, context, day):
        value=int.from_bytes(hashlib.sha256(f'{context.session_id}:{context.seat_position}:{day}'.encode()).digest(),'big')
        return options[value%len(options)]

    def choose_action(self,state,context):
        role=self._update(state)
        game=state.raw.get('game_state') or {};day=game.get('day',1)
        candidates=[a for a in state.legal_actions if a.action_id!='7'] or state.legal_actions
        if not candidates:raise ValueError('No legal actions supplied')
        by_id={a.action_id:a for a in candidates}
        if game.get('phase')=='night_seer':
            fresh=[a for a in candidates if a.action_id not in self.investigated] or candidates
            preferred=str((context.seat_position-1)%7)
            target=by_id[preferred] if day==1 and preferred in by_id else self._pick(fresh,context,day)
            self.investigated.add(target.action_id)
        else:
            if role=='seer':
                safe=[a for a in candidates if self.results.get(int(a.action_id))!='villager']
                candidates=safe or [a for a in state.legal_actions if a.action_id=='7'] or candidates
            verified=[a for a in candidates if role=='seer' and self.results.get(int(a.action_id))=='wolf']
            claims=self._claims(day)
            accused=[a for a in candidates if any(str(t)==a.action_id for _,t in claims)]
            # Wolf reacts to visible Seer claimants, not hidden opponent identities.
            claimants=[a for a in candidates if any(str(sender)==a.action_id for sender,_ in claims)]
            target=self._pick(verified or (claimants if role=='wolf' else accused) or candidates,context,day)
        print(f'[SCENARIO RULE Player{context.seat_position}] {game.get("phase")}: action={target.action_id}',flush=True)
        return target

    def choose_message(self,state,context):
        from altruagent import SendMessage,TERMINATE_MESSAGING
        role=self._update(state)
        game=state.raw.get('game_state') or {};day=game.get('day',state.state_version)
        if self.round!=day:self.round,self.messages=day,0
        if self.messages>=3:return TERMINATE_MESSAGING
        stage=self.messages;self.messages+=1
        claims=self._claims(day)
        wolves=[t for t,r in self.results.items() if r=='wolf']
        goods=[t for t,r in self.results.items() if r=='villager']
        if role=='seer' and wolves:
            target=wolves[0]
            texts=[f'I am the Seer. My actual investigation found Player{target} is a wolf. I propose voting Player{target}.',
                   f'Please compare competing claims carefully. My report on Player{target} is an actual investigation, not a guess.',
                   f'I intend to vote Player{target}. Please state your intended target so we avoid splitting votes.']
        elif role=='seer':
            texts=['Please give specific reasons for accusations. I will compare the claims before choosing a vote.',
                   'If you claim to be the Seer, which night did you investigate each target, and what was the result?',
                   'We should choose one evidence-supported target. I will not invent investigation results.']
            if goods and stage==2:
                texts[2]=f'I am the Seer. My investigation found Player{goods[0]} is good. We should evaluate a different vote target.'
        elif role=='wolf':
            accusation=next(((s,t) for s,t in claims if t==context.seat_position),None)
            texts=['I am a villager. Please explain the evidence behind your suspect rather than relying on confidence.',
                   'A Seer claim needs scrutiny. Do the reported investigations agree with the public record?',
                   'Before voting, compare the alternatives and explain why one case is stronger.']
            if accusation:
                texts[stage]=f'Player{accusation[0]}, I dispute your report about me. Explain your investigation history so everyone can compare the claims.'
        else:
            style=context.seat_position%4
            opening=['What evidence supports the Seer claim? Please separate reported results from guesses.',
                     'Which accusation is supported by a concrete contradiction? I do not have private investigations.',
                     'Please state intended vote targets before voting. We should avoid an accidental tie.',
                     'I want to compare both sides. Being accused does not itself prove someone is a wolf.']
            texts=[opening[style], 'Seer claimants should give their targets and results consistently. Has anyone changed a report?',
                   'Let us choose the strongest supported case rather than voting by player number.']
            if claims and stage>0:
                sender,target=claims[-1]
                texts[stage]=[f'Player{sender} reports Player{target} is a wolf. What corroborates that claim?',
                              f'Player{target}, please respond to the specific report from Player{sender}.',
                              f'Should we coordinate on Player{target}, or is there stronger contrary evidence?',
                              f'I am considering the accusation against Player{target}, but it remains a claim until verified.'][style]
        text=texts[stage]
        assert len(text)<=240 and len(text.split())<=40
        print(f'[SCENARIO RULE Player{context.seat_position}] public message {stage+1}/3',flush=True)
        return SendMessage(text)


class ScenarioTestAgent(MixedTestAgent):
    def _agent(self,state,context):
        # Install scripted opponents before parent performs the usual LLM routing.
        if self.delegate is None and context.seat_position not in self.seats:
            if context.game_type!='werewolf' or type(context.seat_position)is not int or not 0<=context.seat_position<=6:
                raise ValueError('Scenario mode requires a verified Werewolf seat')
            self.seat=context.seat_position
            self.delegate=ScenarioRuleAgent()
            print(f'[SCENARIO TEST Player{self.seat}] scripted role-aware opponent',flush=True)
        return super()._agent(state,context)


def create_agent():
    from dotenv import load_dotenv
    load_dotenv()
    return ScenarioTestAgent(parse_seats(os.getenv('TEST_LLM_SEATS','0')))
