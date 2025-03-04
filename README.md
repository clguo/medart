## Med-Art: Diffusion Transformer for Text-to-Image Generation in Medical Domain<br><sub>PyTorch Implementation</sub>



![Med-Art Samples](img/results.png)

This repo features a PyTorch implementation for the paper [**Med-Art: Diffusion Transformer for Text-to-Image Generation in Medical Domain**]

It contains:


* ⚡️ Pre-trained class-conditional DiT models trained on ImageNet (512x512 and 256x256)
* 💥 A self-contained [Hugging Face Space](https://huggingface.co/spaces/wpeebles/DiT) and [Colab notebook](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb) for running pre-trained DiT-XL/2 models
* 🛸 An improved DiT [training script](train.py) and several [training options](train_options)

## Setup

First, download and set up the repo:

```bash
git clone https://anonymous.4open.science/r/medart-4986
cd medart
```

```bash
conda create -n medart python=3.10 
conda activate medart
```
* 🪐 Installation of **Llava-Next** and its dependencies, along with code for **image-to-text generation (Visual Symptom Generator) [implementation](models.py)**
  Ensure installation is performed in a GPU environment.
```bash
cd medart/LLaVA-NeXT
pip install -e ".[train]"
pip install flash-attn==2.4.1
```
Run VSG to generate visual descriptions of images.
```bash
python VSG.py
```
Save the generated CSV as metadata.csv (with the first column as file_name and the second column as text) under the train folder.

* 💥 Install diffusers.
Do not use the official diffusers here, as it is necessary to ensure that DPM Solver++ supports backpropagation.
```bash
cd medart/diffusers
pip install .
```

## Training
The weights from PixArt-α will be downloaded automatically.
```bash
dataset_id="dataset/kvasir/train"
model_id=PixArt-alpha/PixArt-XL-2-512x512
accelerate launch  --mixed_precision="bf16" --num_processes=1 --main_process_port=36667  medart.py \
  --pretrained_model_name_or_path=$model_id \
  --dataset_name=$dataset_id \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=15 \
  --rank=8 \
  --use_8bit_adam \
  --checkpointing_steps=12800 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="medart" \
  --gradient_checkpointing \
  --checkpoints_total_limit=10 \
  --max_token_length=120\
  --M_times=20\
  --N_steps=500 
```


```bash
python sample.py --image-size 512 --seed 1
```

For convenience, our pre-trained DiT models can be downloaded directly here as well:

| DiT Model     | Image Resolution | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) | 256x256          | 2.27    | 278.24          | 119    |
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) | 512x512          | 3.04    | 240.82          | 525    |


**Custom DiT checkpoints.** If you've trained a new DiT model with [`train.py`](train.py) (see [below](#training-dit)), you can add the `--ckpt`
argument to use your own checkpoint instead. For example, to sample from the EMA weights of a custom 
256x256 DiT-L/4 model, run:

```bash
python sample.py --model DiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```


## Training
### Preparation Before Training
To extract ImageNet features with `1` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 --data-path /path/to/imagenet/train --features-path /path/to/store/features
```

### Training DiT
We provide a training script for DiT in [`train.py`](train.py). This script can be used to train class-conditional 
DiT models, but it can be easily modified to support other types of conditioning. 

To launch DiT-XL/2 (256x256) training with `1` GPUs on one node:

```bash
accelerate launch --mixed_precision fp16 train.py --model DiT-XL/2 --features-path /path/to/store/features
```

To launch DiT-XL/2 (256x256) training with `N` GPUs on one node:
```bash
accelerate launch --multi_gpu --num_processes N --mixed_precision fp16 train.py --model DiT-XL/2 --features-path /path/to/store/features
```

Alternatively, you have the option to extract and train the scripts located in the folder [training options](train_options).


### PyTorch Training Results

We've trained DiT-XL/2 and DiT-B/4 models from scratch with the PyTorch training script
to verify that it reproduces the original JAX results up to several hundred thousand training iterations. Across our experiments, the PyTorch-trained models give 
similar (and sometimes slightly better) results compared to the JAX-trained models up to reasonable random variation. Some data points:

| DiT Model  | Train Steps | FID-50K<br> (JAX Training) | FID-50K<br> (PyTorch Training) | PyTorch Global Training Seed |
|------------|-------------|----------------------------|--------------------------------|------------------------------|
| XL/2       | 400K        | 19.5                       | **18.1**                       | 42                           |
| B/4        | 400K        | **68.4**                   | 68.9                           | 42                           |
| B/4        | 400K        | 68.4                       | **68.3**                       | 100                          |

These models were trained at 256x256 resolution; we used 8x A100s to train XL/2 and 4x A100s to train B/4. Note that FID 
here is computed with 250 DDPM sampling steps, with the `mse` VAE decoder and without guidance (`cfg-scale=1`). 


### Improved Training Performance
In comparison to the original implementation, we implement a selection of training speed acceleration and memory saving features including gradient checkpointing, mixed precision training, and pre-extracted VAE features, resulting in a 95% speed increase and 60% memory reduction on DiT-XL/2. Some data points using a global batch size of 128 with an A100:
 
| gradient checkpointing | mixed precision training | feature pre-extraction | training speed | memory       |
|:----------------------:|:------------------------:|:----------------------:|:--------------:|:------------:|
| ❌                    | ❌                       | ❌                    | -              | out of memory|
| ✔                     | ❌                       | ❌                    | 0.43 steps/sec | 44045 MB     |
| ✔                     | ✔                        | ❌                    | 0.56 steps/sec | 40461 MB     |
| ✔                     | ✔                        | ✔                     | 0.84 steps/sec | 27485 MB     |


## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a DiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained DiT-XL/2 model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

There are several additional options; see [`sample_ddp.py`](sample_ddp.py) for details.


## Citation

```bibtex
@misc{jin2024fast,
    title={Fast-DiT: Fast Diffusion Models with Transformers},
    author={Jin, Chuanyang and Xie, Saining},
    howpublished = {\url{https://github.com/chuanyangjin/fast-DiT}},
    year={2024}
}
```
