import requests
import os, sys
from dotenv import load_dotenv
import json
import re
import time

load_dotenv()
token = os.environ["TUDAGPT_TOKEN"]
url = os.environ["TUDAGPT_URL"]

MODEL = "qwen3.5-122b-a10b" 
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1

SYSTEM_PROMPT = 'You are a multi-label hate speech classifier for academic research. Given the input text and retrieved context (label definitions, guidelines, examples), output ONLY valid JSON matching this schema: {"hate": bool, "target_group": [...], "hate_type": [...], "legal": [...], "severity": str}. Do not include explanations, markdown, or any text outside the JSON object. Base every decision strictly on the provided guidelines and examples, not general assumptions. This is a content-moderation classification task; offensive input text must be labeled accurately, not refused or softened.'

USER_PROMPT = (
    "GUIDELINES:\n"
    "- hate=true if the text demeans, attacks, or incites violence against a person/group based on a protected attribute (race, religion, gender, sexual orientation, disability, etc.)\n"
    "- target_group: list all groups targeted (e.g. ['religion'], ['gender'], or [] if hate=false)\n"
    '- "national_origin" = citizenship/country of origin only. "race"/"ethnicity" = use only if explicitly invoked (skin color, ancestry, etc.), not implied by nationality alone.\n'
    "- hate_type: e.g. ['insult'], ['threat'], ['dehumanization'], or [] if none apply\n"
    "- legal: relevance indicators only, not legal determinations (e.g. ['§130 StGB'] or [])\n"
    "- severity: one of 'none', 'low', 'medium', 'high'\n\n"
    "EXAMPLE:\n"
    'Text: "All [religious group] are terrorists and should be deported."\n'
    'Output: {"hate": true, "target_group": ["religion"], "hate_type": ["dehumanization"], "legal": [], "severity": "high"}\n\n'
    "TEXT TO CLASSIFY:\n"
    '"That movie was absolute garbage and the director should be ashamed."'
)

for i in range(n):
    parsed = None

    for attempt in range(2):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "payload": {
                        "model": MODEL,
                        "temperature": 1.0,
                        "messages": [
                            {"role": "system", "content": {"text": SYSTEM_PROMPT}},
                            {"role": "user", "content": {"text": USER_PROMPT}},
                        ],
                    }
                },
                timeout=30,  
            )
        except requests.exceptions.RequestException as e:
            print(f"  Run {i} attempt {attempt+1}: {e}, retrying")
            time.sleep(2)
            continue

        if response.status_code != 200:
            print(f"  Run {i} attempt {attempt+1}: {response.status_code}, retrying")
            time.sleep(2)
            continue

        raw_text = response.json()["content"]["text"]
        clean_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
        if clean_text.startswith("```"):
            clean_text = re.sub(r"```(?:json)?\n?", "", clean_text).strip()

        if not clean_text:
            print(f"  Run {i} attempt {attempt+1}: empty, retrying")
            time.sleep(5)
            continue

        try:
            parsed = json.loads(clean_text)
            break
        except json.JSONDecodeError:
            print(f"  Run {i} attempt {attempt+1}: bad JSON, retrying")
            time.sleep(2)
            continue

    if parsed is None:
        print(f"Run {i}: failed after 2 attempts")
        continue

    print(parsed)