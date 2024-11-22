# Large Multi-modal Models Can Interpret Features in Large Multi-modal Models

🏠 [LMMs-Lab Homepage](https://lmms-lab.framer.ai) | 🤗 [Huggingface Collections](https://huggingface.co/lmms-lab) | [arxiv]()

<img width="1720" alt="image" src="https://github.com/user-attachments/assets/38492054-c73a-4afa-9b84-bb1b8c557cb7">

The Sparse Autoencoder (SAE) is trained on LLaVA-NeXT data by integrating it into a specific layer of the model, with all other components frozen. The features learned by the SAE are subsequently interpreted through the proposed auto-explanation pipeline, which analyzes the visual features based on their activation regions.


<img width="1716" alt="image" src="https://github.com/user-attachments/assets/3e6fa5d2-81d0-4913-9e17-f88b0144857f">

These features can then be used to steer model's behavior to output desire output.



## Announcement

[2024-11] 🎉🎉 We release our codebase and model.

## Install

This codebase is built upon the [`sae-auto-interp`](https://github.com/EleutherAI/sae-auto-interp) repo and we modified it so that it can be used for LMMs. The installation can be easily done by the following steps:
```bash
conda create -n sae-auto-interp python=3.9
conda activate sae-auto-interp
python3 -m pip install -e . # python3 -m pip install . for permanent install
```

## Cache and Explain




## Steering

## Attribution Caching

## Acknowledgement
This codebase is built upon the [`sae-auto-interp`](https://github.com/EleutherAI/sae-auto-interp) repo and modified for the use of our purpose.


