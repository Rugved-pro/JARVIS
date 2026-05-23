"""
Brain Module
============
Interfaces with a local Ollama server running llama3.1.
Sends user messages and streams back LLM responses.
Maintains multi-turn conversation history within a session.
"""

import json
import logging
from typing import Generator

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class OllamaBrain:
    """
    The 'brain' of Jarvis — sends prompts to a local Ollama instance
    and yields streamed response chunks for low-latency TTS piping.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._system_prompt = system_prompt
        self._conversation: list[dict] = []

        logger.info(f"OllamaBrain initialized — model='{model}', url='{base_url}'")

    def _check_connection(self) -> bool:
        """Verify that the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def think(self, user_message: str) -> Generator[str, None, None]:
        """
        Send a user message to Ollama and yield response text chunks
        as they stream in.

        Args:
            user_message: The transcribed user speech.

        Yields:
            String chunks of the assistant's response.

        Raises:
            ConnectionError: If Ollama is not reachable.
        """
        if not user_message.strip():
            return

        # Build the messages payload
        self._conversation.append({
            "role": "user",
            "content": user_message,
        })

        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._conversation,
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }

        logger.info(f"Sending to Ollama: '{user_message[:80]}...'")

        full_response = []

        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract the content chunk from the streamed response
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    full_response.append(chunk)
                    yield chunk

                # Check if the response is complete
                if data.get("done", False):
                    break

            # Store the full assistant response in conversation history
            assistant_text = "".join(full_response)
            self._conversation.append({
                "role": "assistant",
                "content": assistant_text,
            })

            logger.info(
                f"Ollama response complete "
                f"({len(assistant_text)} chars, "
                f"{len(self._conversation) // 2} turns)"
            )

        except requests.ConnectionError:
            error_msg = (
                "Cannot connect to Ollama. "
                "Make sure it's running: 'ollama serve'"
            )
            logger.error(error_msg)
            yield error_msg

        except requests.Timeout:
            error_msg = "Ollama request timed out. The model may be loading."
            logger.error(error_msg)
            yield error_msg

        except requests.HTTPError as e:
            error_msg = f"Ollama returned an error: {e}"
            logger.error(error_msg)
            yield error_msg

    def reset_conversation(self) -> None:
        """Clear conversation history to start fresh."""
        self._conversation.clear()
        logger.info("Conversation history cleared.")

    @property
    def turn_count(self) -> int:
        """Number of user-assistant turn pairs in current conversation."""
        return len(self._conversation) // 2
