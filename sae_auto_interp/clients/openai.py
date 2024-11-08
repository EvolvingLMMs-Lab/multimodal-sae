import json
from time import sleep
from typing import Dict, List, Literal, Union

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .client import Client


class OpenAIClient(Client):
    def __init__(
        self,
        model: str,
        api_type: Literal["openai", "azure"],
        api_endpoint: str,
        api_version: str = None,
        api_key: str = None,
        deployment_name: str = None,
        timeout: int = 600,
    ):
        super().__init__(model)
        if api_type == "openai":
            self.client = AsyncOpenAI(base_url=api_endpoint, api_key=api_key)
        elif api_type == "azure":
            self.client = AsyncAzureOpenAI(
                azure_endpoint=api_endpoint,
                api_key=api_key,
                azure_deployment=deployment_name,
                api_version=api_version,
            )

        self.timeout = timeout

    async def generate(
        self,
        prompt: Union[str, List[Dict]],
        raw: bool = False,
        temperature: float = 0,
        max_retries: int = 2,
        **kwargs,
    ) -> str:
        messages = []
        content = []
        if isinstance(prompt, str):
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages = prompt

        try:
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        timeout=self.timeout,
                        **kwargs,
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
