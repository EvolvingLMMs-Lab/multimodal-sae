import json
from asyncio import sleep

from openai import AsyncOpenAI

from ..logger import logger
from .client import Client


class Local(Client):
    provider = "vllm"

    def __init__(self, model: str, base_url="http://localhost:8000/v1"):
        super().__init__(model)
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=None)
        self.model = model

    async def generate(
        self,
        prompt: str,
        raw: bool = False,
        use_legacy_api: bool = False,
        max_retries: int = 2,
        **kwargs,
    ) -> str:
        """
        Wrapper method for vLLM post requests.
        """
        try:
            for attempt in range(max_retries):
                try:
                    if use_legacy_api:
                        response = await self.client.completions.create(
                            model=self.model, prompt=prompt, **kwargs
                        )
                    else:
                        response = await self.client.chat.completions.create(
                            model=self.model, messages=prompt, **kwargs
                        )
                    if response is None:
                        raise Exception("Response is None")
                    return response if raw else self.postprocess(response)

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Attempt {attempt + 1}: Invalid JSON response, retrying... {e}"
                    )

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: {str(e)}, retrying...")

                await sleep(1)
        except Exception as e:
            logger.error(f"All retry attempts failed. Most recent error: {e}")
            raise

    def postprocess(self, response: dict) -> str:
        """
        Postprocess the response from the API.
        """
        return response.choices[0].message.content
