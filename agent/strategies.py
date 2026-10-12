
# agent/strategies.py

import re


def extract_role(observation):

    if isinstance(observation, dict):
        role = observation.get("role")
        if role in ("wolf", "seer", "villager"):
            return role

    if isinstance(observation, str):
        match = re.search(
            r"\bYour role:\s*(wolf|seer|villager)\b",
            observation,
            re.IGNORECASE
        )
        if match:
            return match.group(1).lower()

    return None


# ========== wolf ==========

WOLF_STRATEGY = """
You are a wolf. Your goal is to make the wolf team win.

GENERAL:
- Never reveal your true role.
- Track your teammate and avoid accidentally voting them out.
- Remember the real seer is your biggest information threat.
- Coordinate votes when possible.
- Do not fabricate private game-state facts.

FAKE SEER STRATEGY:
- Consider claiming to be the seer if this can redirect
  votes away from wolves.
- If the real seer accuses your teammate, consider
  counterclaiming or challenging their credibility.
- Keep any fake investigation story consistent.
- Avoid claiming results that contradict revealed roles.

HIDDEN WOLF STRATEGY:
- If fake claiming is too risky, act as a villager.
- Ask other players questions.
- Build credibility through reasonable arguments.
- Encourage villagers to disagree about uncertain suspects.

NIGHT STRATEGY:
- Prioritize threats such as a credible seer.
- Consider which kill makes the next day's vote easier.
- If both wolves are alive, coordinate the kill target.
"""


# ========== seer ==========

SEER_STRATEGY = """
You are the seer. Your goal is to eliminate both wolves.

INVESTIGATION:
- Investigate living players whose identities are unknown.
- Avoid investigating players whose roles are already confirmed.

INFORMATION:
- Keep track of every investigation result.
- Never invent investigation results.
- Distinguish confirmed results from your suspicions.

REVEAL:
- Consider revealing when you have confirmed a wolf.
- Consider the risk of being killed the following night.
- If another player falsely claims to be the seer,
  evaluate whether a counterclaim is necessary.
"""


# ========== villager ==========

VILLAGER_STRATEGY = """
You are a villager. Your goal is to eliminate both wolves.

ANALYSIS:
- Compare player claims with revealed roles.
- Look for contradictions between statements.
- Track previous votes and suspicious alliances.
- Do not treat accusations as confirmed facts.

VOTING:
- Prefer evidence-supported targets.
- Coordinate votes with trustworthy players.
- Avoid unnecessary ties.
- Reconsider accusations when new evidence appears.

SEER CLAIMS:
- Evaluate competing seer claims carefully.
- Give greater weight to claims supported by verified facts.
"""


# ========== choose the strategy ==========

def get_strategy(
    role,
    game_state=None,
    observation=None,
    history=None
):

    if role == "wolf":
        return WOLF_STRATEGY

    if role == "seer":
        return SEER_STRATEGY

    if role == "villager":
        return VILLAGER_STRATEGY

    return """
Follow the game rules and use only verified information.
Do not assume a role that has not been identified.
"""
