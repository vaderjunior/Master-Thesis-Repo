
import os
import requests
from dotenv import load_dotenv

load_dotenv()


class TUDaGPTClient:
    def __init__(self, model: str, temperature: float = 1.0, timeout: int = 120):
        self.token = os.environ["TUDAGPT_TOKEN"]
        self.url = os.environ["TUDAGPT_URL"]  # full endpoint, incl. /api/ai-req
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"/"user", "text": "..."}]
        Returns the raw model output string."""
        body = {
            "payload": {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": m["role"], "content": {"text": m["text"]}}
                    for m in messages
                ],
            }
        }

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["content"]["text"]