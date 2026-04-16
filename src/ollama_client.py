import json
import requests


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return True, "Ollama reachable"
        except Exception as exc:
            return False, f"Ollama unavailable: {exc}"

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": inputs},
            timeout=self.timeout,
        )

        if response.status_code == 200:
            data = response.json()
            return data["embeddings"]

        if response.status_code == 404:
            embeddings = []
            for text in inputs:
                fallback = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=self.timeout,
                )
                fallback.raise_for_status()
                data = fallback.json()

                if "embedding" in data:
                    embeddings.append(data["embedding"])
                elif "embeddings" in data and data["embeddings"]:
                    embeddings.append(data["embeddings"][0])
                else:
                    raise ValueError(f"Unexpected embeddings response format: {data}")
            return embeddings

        response.raise_for_status()
        return []

    def chat(self, model: str, messages: list[dict], keep_alive: str = "10m") -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": keep_alive,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    def stream_chat(self, model: str, messages: list[dict], keep_alive: str = "10m"):
        with requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "keep_alive": keep_alive,
            },
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                payload = json.loads(line.decode("utf-8"))
                msg = payload.get("message", {})
                content = msg.get("content", "")
                if content:
                    yield content