### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."""


GUIDELINES = """Guidelines:

You will be given a list of image examples on which the neuron activates. The region where the neuron activates on has been highlighted and the other regions are masked out. These images may or may not share similar visual patterns that related to the feature that is activated.

Please only caption on the region where the images are being highlighted. It is possibly that the neuron is not activated on the images at all or the activated features is pretty small.

Your task is to
- Try to produce a concise final description. Simply describe the features that activate the neuron, and what its role might be based on the images.
- Please output your description in a concise description way, such as \"The train\", \"The sunglasses\", that directly depicts the object where the region is activated. If it is hard to use just a single word, please describe it using a short phrase, such as \"The word A on the trafic sign\".
- When output the explanations, please first output \"[EXPLANATION]:\" and then follows your explanations.
- The last line of your response must be the formatted explanation.
- If it is unable to determine the shared visual features, you can simply output \"[EXPLANATION]: Unable to produce descriptions\"
"""


def build_prompt(
    images,
):
    messages = [
        {"role": "system", "content": SYSTEM},
    ]
    content = [{"type": "text", "text": GUIDELINES}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"},
                "modalities": "multi-images",
            }
        )
    messages.append({"role": "user", "content": content})
    return messages
