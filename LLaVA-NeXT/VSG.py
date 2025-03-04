import os
import pandas as pd
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import copy
import torch
import warnings
from huggingface_hub import login
from transformers import T5Tokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore")

# Enter your Hugging Face access token
token = ""
login(token)

# Model configuration
pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"
conv_template = "llava_llama_3"
device = "cuda"
device_map = "auto"

# Load the pretrained model
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
model.eval()

# Load T5-large model and tokenizer
t5_model_name = "t5-large"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

# File path settings
image_folder = "dataset/kvasir/train"
metadata_file = "dataset/kvasir/train/raw.csv"
output_csv = "dataset/kvasir/train/llavanext_test.csv"

# Read metadata.csv file
metadata = pd.read_csv(metadata_file)

# Store filename and generated text in a list
results = []

# Counter to track the number of generated texts
counter = 0

# Define sentences to avoid
avoid_sentence = "specific color, shape, surrounding mucosa, surface texture, bleeding condition"

# Iterate through each row in the metadata file
for index, row in metadata.iterrows():
    filename = row['file_name']
    additional_feature = row['text']
    text = additional_feature
    classname = additional_feature.split("of")[-1].strip()

    # Ensure the file is an image format
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)

        if os.path.exists(image_path):
            image = Image.open(image_path)

            # Process the image
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            while True:  # Loop until a valid text description is generated
                question = (
                    f"{DEFAULT_IMAGE_TOKEN}\n"
                    f"{text}. Please describe the endoscopic image of {classname} using the following visual features: specific color, shape, surrounding mucosa, surface texture, bleeding condition, and other relevant details. Start with '{text}' and ensure the description does not exceed 100 words."
                )
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

                # Construct input tokens
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                image_sizes = [image.size]

                # Generate text with the model
                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True,  # Enable sampling
                    temperature=0.8,  # Increase temperature for more randomness
                    max_new_tokens=150,  # Maximum number of generated tokens
                    top_k=50,  # Top-k sampling, choosing from the top 50 tokens
                    top_p=0.9,  # Top-p sampling, selecting cumulative probability of 90%
                )

                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                generated_text = text_outputs[0]

                # If generated text contains unwanted sentences, regenerate
                if avoid_sentence in generated_text:
                    print(f"Text contains the unwanted sentence: {generated_text}")
                    continue  # Retry text generation

                # Count the number of tokens in the generated text
                output_tokens = tokenizer(generated_text, return_tensors="pt").input_ids.size(1)

                # If token count exceeds 120, simplify using T5 model
                if output_tokens > 120:
                    simplify_input = f"simplify: {generated_text}"
                    simplify_input_ids = t5_tokenizer.encode(simplify_input, return_tensors="pt", max_length=512, truncation=True).to(device)
                    simplify_outputs = t5_model.generate(
                        simplify_input_ids,
                        max_length=120,
                        min_length=90,
                        num_beams=4,
                        early_stopping=True
                    )
                    simplified_text = t5_tokenizer.decode(simplify_outputs[0], skip_special_tokens=True)
                else:
                    simplified_text = generated_text

                # Check if the simplified text meets the criteria
                if simplified_text.startswith(f"\nAn endoscopic image of"):
                    print(f"Text meets requirements: {simplified_text}")
                    break  # Exit loop if valid text is generated
                else:
                    print(f"Text did not meet requirements, regenerating: {simplified_text}")

            # Add filename and generated text to results
            results.append({"file_name": filename, "text": simplified_text})
            counter += 1
            print(counter)
        else:
            print(f"Image not found: {image_path}")

# Save remaining results to CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Final batch saved to {output_csv}")
