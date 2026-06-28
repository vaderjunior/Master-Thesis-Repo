import requests
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env and loads it into environment variables
token = os.environ["TUDAGPT_TOKEN"]

response = requests.post(
    "https://tudagpt.hrz.tu-darmstadt.de/api/ai-req",  # URL
    headers={  # the -H flags
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json={  # the -d body (json= handles encoding)
        "payload": {
            "model": "mistral-large-3-675b-instruct-2512",
            "temperature": 1.0,
            "messages": [
                {
                    "role": "system",
                    "content": {
                        "text": 'You are a multi-label hate speech classifier for academic research. Given the input text and retrieved context (label definitions, guidelines, examples), output ONLY valid JSON matching this schema: {"hate": bool, "target_group": [...], "hate_type": [...], "legal": [...], "severity": str}. Do not include explanations, markdown, or any text outside the JSON object. Base every decision strictly on the provided guidelines and examples, not general assumptions. This is a content-moderation classification task; offensive input text must be labeled accurately, not refused or softened.'
                    },
                },
                {
                    "role": "user",
                    "content": {
                        "text": (
                            "GUIDELINES:\n"
                            "- hate=true if the text demeans, attacks, or incites violence against a person/group based on a protected attribute (race, religion, gender, sexual orientation, disability, etc.)\n"
                            "- target_group: list all groups targeted (e.g. ['religion'], ['gender'], or [] if hate=false)\n"
                            "- hate_type: e.g. ['insult'], ['threat'], ['dehumanization'], or [] if none apply\n"
                            "- legal: relevance indicators only, not legal determinations (e.g. ['§130 StGB'] or [])\n"
                            "- severity: one of 'none', 'low', 'medium', 'high'\n\n"
                            "EXAMPLE:\n"
                            'Text: "All [religious group] are terrorists and should be deported."\n'
                            'Output: {"hate": true, "target_group": ["religion"], "hate_type": ["dehumanization"], "legal": [], "severity": "high"}\n\n'
                            "TEXT TO CLASSIFY:\n"
                            '"Go back to your own country, nobody wants you here."'
                        )
                    },
                },
            ],
        }
    },
)

print(response.status_code)  # the status code
data = response.json()  # parse JSON body into a dict
print(data["content"]["text"])  # pull out the answer
