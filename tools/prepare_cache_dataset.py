import argparse

from datasets import concatenate_datasets, load_dataset

datasets_path = [
    "lmms-lab/LLaVA-Bench-Wilder",
    "lmms-lab/MME",
    "lmms-lab/COCO-Caption2017",
    "lmms-lab/MMVet",
    "lmms-lab/LLaVA-NeXT-Data",
]

datasets_split = ["test", "test", "val", "test", "train[:5%]"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    first_dataset = load_dataset(datasets_path[0], split=datasets_split[0])
    first_dataset = first_dataset.select_columns(["image"])
    source = [datasets_path[0]] * len(first_dataset)
    for dataset_path, dataset_split in zip(datasets_path[1:], datasets_split[1:]):
        dataset = load_dataset(dataset_path, split=dataset_split)
        dataset = dataset.select_columns(["image"])
        first_dataset = concatenate_datasets([first_dataset, dataset])
        source += [dataset_path] * len(dataset)
    first_dataset = first_dataset.add_column("source", source)
    first_dataset.push_to_hub(args.output_path)
