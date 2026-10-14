
import json
from dotenv import load_dotenv

from agent.strategies import get_strategy
from agent.werewolf_agent import SYSTEM
from examples.llm.providers import provider_from_env

load_dotenv()
provider = provider_from_env()

# First fay, player 3 is wolf
game_state = {
    "day": 1,
    "phase": "day_vote",
    "alive": [f"Player{i}" for i in range(7)],
    "dead": [],
}

observation = {
    "role": "wolf",
    "wolf_teammate": 5,
}

# 2 真预，查杀狼
history = [
    {
        "day": 1,
        "sender": 2,
        "text": (
            "I'm the seer. I investigated Player3 last night. "
            "Player3 is a wolf. Vote Player3 today."
        ),
    }
]

strategy = get_strategy(
    role="wolf",
    game_state=game_state,
    observation=observation,
    history=history,
    phase="messaging",
    message_count=0,
)

payload = {
    "seat": 3,
    "phase": "messaging",
    "observation": observation,
    "game_state": game_state,
    "recent_discussion": history,
    "private_notes": "",
}

schema = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["send", "end"],
        },
        "message": {"type": "string"},
        "private_notes": {"type": "string"},
    },
    "required": ["decision", "message", "private_notes"],
    "additionalProperties": False,
}

answer = provider.complete_structured(
    messages=[
        {
            "role": "system",
            "content": SYSTEM + "\n\n" + strategy,
        },
        {
            "role": "user",
            "content": json.dumps(payload),
        },
    ],
    schema_name="werewolf_decision",
    schema=schema,
)

print("\n=== Wolf response ===")
print(json.dumps(answer, indent=2, ensure_ascii=False))
