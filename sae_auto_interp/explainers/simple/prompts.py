### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior.
{prompt}
Guidelines:

You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like <<this>>. If a sequence of consecutive tokens all cause the neuron to activate, the entire sequence of tokens will be contained between delimiters <<just like this>>. The activation value of the example is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text features that activate the neuron, and what its role might be based on the tokens it predicts.
- If either the text features or the predicted tokens are completely uninformative, you don't need to mention them.
- The last line of your response must be the formatted explanation."""

COT = """
(Part 1) Tokens that the neuron activates highly on in text

Step 1: List a couple activating and contextual tokens you find interesting. Search for patterns in these tokens, if there are any. Don't list more than 5 tokens.
Step 2: Write down general shared features of the text examples.
"""

ACTIVATIONS = """
(Part 1) Tokens that the neuron activates highly on in text

Step 1: List a couple activating and contextual tokens you find interesting. Search for patterns in these tokens, if there are any.
Step 2: Write down several general shared features of the text examples.
Step 3: Take note of the activation values to understand which examples are most representative of the neuron.
"""

LOGITS = """
(Part 2) Tokens that the neuron boosts in the next token prediction

You will also be shown a list called Top_logits. The logits promoted by the neuron shed light on how the neuron's activation influences the model's predictions or outputs. Look at this list of Top_logits and refine your hypotheses from part 1. It is possible that this list is more informative than the examples from part 1.

Pay close attention to the words in this list and write down what they have in common. Then look at what they have in common, as well as patterns in the tokens you found in Part 1, to produce a single explanation for what features of text cause the neuron to activate. Propose your explanation in the following format:
[EXPLANATION]: <your explanation>
"""

### EXAMPLE 1 ###

EXAMPLE_1 = """
Example 1:  and he was <<over the moon>> to find
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
"""

EXAMPLE_1_ACTIVATIONS = """
Example 1:  and he was <<over the moon>> to find
Activations: ("over the moon", 9)
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till the cows come home", 5)
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than meets the eye", 8)
"""

EXAMPLE_1_LOGITS = """
Top_logits: ["elated", "joyful", "story", "thrilled", "spider"]
"""

### EXAMPLE 1 RESPONSE ###

EXAMPLE_1_COT_RESPONSE = """
(Part 1)
ACTIVATING TOKENS: "over the moon", "than meets the eye".
PREVIOUS TOKENS: No interesting patterns.

Step 1.
The activating tokens are all parts of common idioms.
The previous tokens have nothing in common.

Step 2.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The text examples all convey positive sentiment.
"""

EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
(Part 1)
ACTIVATING TOKENS: "over the moon", "than meets the eye".
PREVIOUS TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all parts of common idioms.
- The previous tokens have nothing in common.

Step 2.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Step 3.
- The activation values are the highest for the more common idioms in examples 1 and 3.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The text examples all convey positive sentiment.
"""

EXAMPLE_1_LOGITS_RESPONSE = """
(Part 2)
SIMILAR TOKENS: "elated", "joyful", "thrilled".
- The top logits list contains words that are strongly associated with positive emotions.
"""

EXAMPLE_1_EXPLANATION = """
[EXPLANATION]: Common idioms in text conveying positive sentiment.
"""

### EXAMPLE 2 ###

EXAMPLE_2 = """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Example 2:  every year you get tall<<er>>," she
Example 3:  the hole was small<<er>> but deep<<er>> than the
"""

EXAMPLE_2_ACTIVATIONS = """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<ish>>," she
Activations: ("ish", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)
"""

EXAMPLE_2_LOGITS = """
Top_logits: ["apple", "running", "book", "wider", "quickly"]
"""

### EXAMPLE 2 RESPONSE ###

EXAMPLE_2_COT_RESPONSE = """
(Part 1)
ACTIVATING TOKENS: "er", "er", "er".
PREVIOUS TOKENS: "wid", "tall", "small", "deep".

Step 1.
- The activating tokens are mostly "er".
- The previous tokens are mostly adjectives, or parts of adjectives, describing size.
- The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

Step 2.
- In each example, the activating token appeared at the end of a comparative adjective.
- The comparative adjectives ("wider", "tallish", "smaller", "deeper") all describe size.

Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed?
- I can't see any.
"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
(Part 1)
ACTIVATING TOKENS: "er", "er", "er".
PREVIOUS TOKENS: "wid", "tall", "small", "deep".

Step 1.
- The activating tokens are mostly "er".
- The previous tokens are mostly adjectives, or parts of adjectives, describing size.
- The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

Step 2.
- In each example, the activating token appeared at the end of a comparative adjective.
- The comparative adjectives ("wider", "tallish", "smaller", "deeper") all describe size.

Step 3.
- Example 2 has a lower activation value. It doesn't compare sizes as directly as the other examples.

Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed?
- I can't see any.
"""

EXAMPLE_2_LOGITS_RESPONSE = """
(Part 2)
SIMILAR TOKENS: None
- The top logits list contains unrelated nouns and adverbs.
"""

EXAMPLE_2_EXPLANATION = """
[EXPLANATION]: The token "er" at the end of a comparative adjective describing size.
"""

### EXAMPLE 3 ###

EXAMPLE_3 = """
Example 1:  something happening inside my <<house>>", he
Example 2:  presumably was always contained in <<a box>>", according
Example 3:  people were coming into the <<smoking area>>".

However he
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
"""

EXAMPLE_3_ACTIVATIONS = """
Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking area", 3)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: (" way?", 2)
"""

EXAMPLE_3_LOGITS = """
Top_logits: ["room", "end", "container, "space", "plane"]
"""

EXAMPLE_3_COT_RESPONSE = """
(Part 1)
ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
PREVIOUS TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all things that one can be in.
- The previous tokens have nothing in common.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The activating token is followed by a quotation mark, suggesting it occurs within speech.
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
(Part 1)
ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
PREVIOUS TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all things that one can be in.
- The previous tokens have nothing in common.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

STEP 3.
- The activation values are highest for the examples where the token is a distinctive object or space.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The activating token is followed by a quotation mark, suggesting it occurs within speech.
"""

EXAMPLE_3_LOGITS_RESPONSE = """
(Part 2)
SIMILAR TOKENS: "room", "container", "space".
- The top logits list suggests a focus on nouns representing physical or metaphorical spaces.
"""

EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Nouns preceding a quotation mark, representing a distinct objects that contains something.
"""


def get(item):
    return globals()[item]


def _prompt(n, logits=False, activations=False, **kwargs):
    starter = (
        get(f"EXAMPLE_{n}") if not activations else get(f"EXAMPLE_{n}_ACTIVATIONS")
    )

    prompt_atoms = [starter]

    if logits:
        prompt_atoms.append(get(f"EXAMPLE_{n}_LOGITS"))

    return "".join(prompt_atoms)


def _response(
    n,
    cot=False,
    logits=False,
    activations=False,
):
    response_atoms = []

    if cot and activations:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_ACTIVATION_RESPONSE"))

    elif cot:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_RESPONSE"))

    if logits:
        response_atoms.append(get(f"EXAMPLE_{n}_LOGITS_RESPONSE"))

    response_atoms.append(get(f"EXAMPLE_{n}_EXPLANATION"))

    return "".join(response_atoms)


def example(n, **kwargs):
    prompt = _prompt(n, **kwargs)
    response = _response(n, **kwargs)

    return prompt, response


def system(
    cot=False,
    logits=False,
    activations=False,
):
    prompt = ""

    if cot and activations:
        prompt += ACTIVATIONS
    elif cot:
        prompt += COT

    if logits:
        prompt += LOGITS

    return [
        {
            "role": "system",
            "content": SYSTEM.format(prompt=prompt),
        }
    ]
