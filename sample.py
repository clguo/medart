import os
import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel
from peft import PeftModel
from PIL import Image

# **Path settings**
ckpt_path = "medart"

# **Initialize PixArt-XL-2 Diffusion Model**
MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"

# Load the LoRA model
transformer = Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.float16)
transformer = PeftModel.from_pretrained(transformer, os.path.join(ckpt_path, "transformer_lora"))

text_encoder = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder = PeftModel.from_pretrained(text_encoder, os.path.join(ckpt_path, "text_encoder_lora"))

# Load the **Diffusion Pipeline**
pipe = PixArtAlphaPipeline.from_pretrained(
    MODEL_ID, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.float16
)
pipe.to("cuda")  # Move the model to GPU for faster inference

# **User input text**
input_text = "An endoscopic image of dyed lifted polyps shows a polyp with a specific blue-green color, indicating it has been dyed. The polyp has a round shape with a smooth surface texture. The surrounding mucosa appears normal with no signs of bleeding. The polyp is lifted and clearly visible, allowing for a thorough examination."  # You can change this text to generate different images

# **Generate an image**
num_inference_steps = 50  # Hyperparameter: Number of inference steps
print(f"Generating image for prompt: {input_text} with {num_inference_steps} steps...")

with torch.no_grad():  # Disable gradient computation for efficiency
    generated_image = pipe(input_text, num_inference_steps=num_inference_steps).images[0]

# **Save the generated image**
output_path = "generated_sample.png"
generated_image.save(output_path)
print(f"Generated image saved at: {output_path}")
