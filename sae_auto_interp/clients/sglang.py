import base64
import json
from asyncio import sleep
from io import BytesIO
from typing import Dict, List, Union

from openai import AsyncOpenAI
from PIL import Image
from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import popen_launch_server

from ..logger import logger
from .client import Client


class SRT(Client):
    """Almost the same as vllm"""

    provider = "sglang"

    def __init__(
        self,
        model: str,
        base_url="http://localhost:8000",
        tp: int = 8,
        # Download the model weights might take
        # very long for the first time
        timeout: int = 180000000,
    ):
        super().__init__(model)
        self.base_url = base_url
        self.model = model
        other_args = []
        other_args.extend(["--tensor-parallel-size", str(tp)])
        if "llava" in model:
            other_args.extend(["--chat-template", "chatml-llava"])
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=timeout,  # First time may
            api_key="EMPTY",
            other_args=other_args,
        )
        self.base_url += "/v1"
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY")

    async def generate(
        self,
        prompt: Union[str, List[Dict]],
        raw: bool = False,
        max_retries: int = 2,
        **kwargs,
    ) -> str:
        """
        Wrapper method for vLLM post requests.
        """
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
                        model=self.model, messages=messages, **kwargs
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

    async def single_image_generate(
        self,
        prompt: str,
        image_path: str,
        raw: bool = False,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_new_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        Wrapper method for vLLM post requests.
        """
        messages = []
        content = []
        images = Image.open(image_path)
        encode_img = self.encode_images(images)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_img}"},
            }
        )
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        try:
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
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

    def encode_images(
        self,
        image: Image,
    ):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def non_async_generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        A Simple test generate function for the server
        """
        messages = []
        content = []
        from openai import OpenAI

        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        new_client = OpenAI(base_url=self.base_url, api_key="EMPTY")
        response = new_client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, **kwargs
        )
        return self.postprocess(response)

    def postprocess(self, response: dict) -> str:
        """
        Postprocess the response from the API.
        """
        return response.choices[0].message.content

    def clean(self):
        kill_child_process(self.process.pid)
