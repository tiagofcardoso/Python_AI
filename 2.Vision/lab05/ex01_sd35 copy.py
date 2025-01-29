# pip install diffusers 
import os
import torch
from diffusers import StableDiffusion3Pipeline

# Set your Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "HUGGINGFACE_TOKEN"

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
)
pipe = pipe.to("cuda")

image = pipe(
    "homer simpson dancing in a disco",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("imagem_ex1.png")
