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
