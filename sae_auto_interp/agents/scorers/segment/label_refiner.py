import asyncio
import json
import os
import re
from multiprocessing import cpu_count
from typing import Dict

from tqdm import tqdm

from sae_auto_interp.clients import SRT

PROMPT = """\
[GUIDELINES]
You are an AI assistant tasked with extracting meaningful labels from descriptions. You will receive a description that may contain references to various entities, and your job is to rephrase and extract the key entities from the text. You will encounter several types of descriptions, and examples for each case are provided below. Please follow the given instructions carefully.

When presenting your answer, first output "[ANSWER]", followed by the extracted entity. Thank you!

Case 1: Good Description
In this case, the description clearly identifies the entity.
Examples:

Description: The cell phone.
Output: [ANSWER] The cell phone

Description: The letters on the shipping containers.
Output: [ANSWER] The letters on the shipping containers

Case 2: Description includes additional words
In this case, the description contains more information than needed. Extract only the key entity.
Examples:

Description: The images all display different models of Honda vehicles, suggesting the neuron is activated by the presence of Honda vehicles or the Honda logo.
Output: [ANSWER] Honda vehicles

Description: The neuron seems to be reacting to the word "ORD" on the billboard. It could be part of a larger word or phrase, but the neuron specifically highlights the letters "ORD." This suggests that the neuron might be specialized in recognizing or processing certain words or characters in images. The activation across the images indicates that the neuron is consistent in its response to textual elements, particularly those that include the "ORD" sequence.
Output: [ANSWER] The word "ORD"

Case 3: Bad Description
In this case, the description does not provide sufficient or valid information.
Examples:

Description: Unable to produce descriptions.
Output: Unable to produce descriptions


[Description]
{description}
"""


class LabelRefiner:
    def __init__(
        self,
        client: SRT,
        features: Dict[str, str],  # Explanation format {feat_name : explain}
    ) -> None:
        self.client = client
        self.features = features

    async def refine(
        self,
    ):
        sem = asyncio.Semaphore(cpu_count() // 8)

        async def _generate(feature_name, prompt):
            async with sem:
                return feature_name, await self.client.generate(prompt)

        pbar = tqdm(total=len(self.features), desc="Refined...")
        self.refine_features = {}
        tasks = [
            asyncio.create_task(
                _generate(feature_name, PROMPT.format(description=explanation))
            )
            for feature_name, explanation in self.features.items()
        ]
        for completed_task in asyncio.as_completed(tasks):
            feature_name, refined_result = await completed_task
            pbar.update(1)
            self.refine_features[feature_name] = self.parse_explanation(refined_result)

    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[ANSWER\]\s*(.*)", text, re.DOTALL)

            return (
                match.group(1).strip()
                if match
                else f"Response {text}. Explanation could not be parsed."
            )
        except Exception:
            return f"Response {text}. Explanation could not be parsed."

    def save_result(self, save_path):
        save_dir = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.refine_features, f, indent=4)
