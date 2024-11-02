### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."""


GUIDELINES = """
[REQUIREMENTS]

1. Focus only on the highlighted region in each image. If no region is highlighted or if the highlighted region is minimal (e.g., a few bright spots), ignore the image.
2. Identify common visual patterns, objects, or concepts in the activated regions. For example, note if highlighted areas show consistent structures, such as mesh patterns or similar objects.

[GUIDELINES]

You will receive a series of images where specific regions have been highlighted to indicate neuron activation. Non-highlighted areas will be masked out or dimmed. Your analysis should consider only the highlighted regions and complete the following tasks:

1. Describe Only the Highlighted Regions: Generate captions solely based on the highlighted regions. If no meaningful pattern is visible, or if only a few scattered spots are highlighted, output: \"[EXPLANATION]: Unable to produce descriptions.\"

2. Concise Description Only: Provide a short, direct description of the common features within the highlighted regions. Avoid any interpretive language—simply state what you see, such as “mesh-like structures” or “actions related to joy or happiness”

3. Output Format: Begin each response with \"[EXPLANATION]:\" followed by your explanation, if applicable. Ensure the last line of your output follows this format.

If unable to determine common visual features, output:

\"[EXPLANATION]: Unable to produce descriptions\"
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
