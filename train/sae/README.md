> [!WARNING]
> The training pipeline of this repo might contains many bugs and is only implement with basic training for a sae with images and chat template format.
> We won't handle errors for this as we haven't developed the training pipeline for a long time.

> [!CAUTION]
> Please use this pipeline with cautions, there might be unexcepted bugs during the training if you modified our training scripts provided

This training pipeline is a detached fork repo on [sae](https://github.com/EleutherAI/sae) from EleutherAI. We recommend you to read the original documention of the repo.

We only made some very minimal changes to support training with chat template and images and may contains many, many bugs in our pipeline.
We currently only support training for 1 batch size and 1 hooked layer and is not recommended if you want to keep scaling the experiment using our pipeline

## Installation
To install this pipeline, you can use the following command:
```bash
cd multimodal-sae/train/sae;
python3 -m pip install -e .
```

## Train

Below is a train script if you want to train a sae with our code

```bash
LLaVA_VERSION="llava-hf/llama3-llava-next-8b-hf"
LLaVA_VERSION_CLEAN="${LLaVA_VERSION//\//_}"
DATASET_VERSION="lmms-lab/LLaVA-NeXT-Data"
DATASET_VERSION_CLEAN="${DATASET_VERSION//\//_}"
echo "Train on $DATASET_VERSION using model $LLaVA_VERSION"

RUN_NAME="$LLaVA_VERSION_CLEAN-$DATASET_VERSION_CLEAN-sae"

torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" -m sae \
    $LLaVA_VERSION \
    $DATASET_VERSION \
    --batch_size 1 \
    --layers 24 \
    --grad_acc_steps 4 \
    --num_latents 131072 \
    --k 256 \
    --run_name $RUN_NAME \
    --micro_acc_steps 1 \
    --data_preprocessing_num_proc 32 \
    --mm_data \
    --split "train"
```

The result training checkpoint can be found [here](https://huggingface.co/lmms-lab/llama3-llava-next-8b-hf-sae-131k)
