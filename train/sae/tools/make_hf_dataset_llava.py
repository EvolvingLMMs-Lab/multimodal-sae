import argparse
import json
import os

from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PIL_Image

NUM_PROC = 32


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path", type=str, help="Path to your instruction file for llava"
    )
    parser.add_argument("--image_folder", type=str, help="Path to your image folder")
    parser.add_argument(
        "--push_to", type=str, help="The repo name that you want to push to"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The subset name if you want for your dataset",
        default=None,
    )
    parser.add_argument(
        "--split", type=str, help="The split name for your dataset", default="train"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Keep your pushed dataset public or not?",
    )

    return parser.parse_args()


def data_generator(data, image_folder):
    for da in data:
        per_df_item = {}
        per_df_item["id"] = str(da["id"])

        # Convert llava format to apply chat template
        # compatible
        new_conversation = []
        for conv in da["conversations"]:
            current_round = {}
            if conv["from"] == "human":
                current_round["role"] = "user"
            elif conv["from"] == "gpt":
                current_round["role"] = "assistant"
            else:
                current_round["role"] = conv["from"]

            current_round["content"] = conv["value"]
            new_conversation.append(current_round)

        per_df_item["conversations"] = new_conversation
        if "image" in da:
            if isinstance(da["image"], list):
                per_df_item["image_path"] = da["image"]
            else:
                per_df_item["image_path"] = [da["image"]]

            # In case that there are multiple images and in list format
            # I feel this is the best way to do so here
            images = []
            image_sizes = []
            for image_path in per_df_item["image_path"]:
                image = PIL_Image.open(os.path.join(image_folder, image_path))
                images.append(image)
                image_sizes.append([image.size[0], image.size[1]])

            per_df_item["image"] = images
            per_df_item["image_sizes"] = image_sizes
        else:
            per_df_item["image_path"] = ""
            per_df_item["image"] = [None]
            per_df_item["image_sizes"] = []

        yield per_df_item


if __name__ == "__main__":
    args = parse_argument()

    dataset_path: str = args.dataset_path
    image_folder: str = args.image_folder
    push_to: str = args.push_to
    dataset_name: str = args.dataset_name
    split: str = args.split
    private: bool = not args.public

    ## Load your data
    with open(dataset_path, "r") as f:
        data = json.load(f)

    features = Features(
        {
            "id": Value("string"),
            # "conversations" : Sequence(
            # feature={"content" : Value("string"),
            # "role" : Value("string")}
            # ),
            "conversations": [
                {"content": Value("string"), "role": Value("string")}
            ],  # This is better than the sequence one
            "image": Sequence(Image()),
            "image_sizes": Sequence(Sequence(Value("int64"))),
        }
    )

    dataset = Dataset.from_generator(
        data_generator,
        gen_kwargs={
            "data": data,
            "image_folder": image_folder,
        },
        num_proc=NUM_PROC,
        features=features,
    )

    dataset.push_to_hub(
        repo_id=push_to,
        config_name="default" if dataset_name is None else dataset_name,
        split=split,
        private=private,
    )
