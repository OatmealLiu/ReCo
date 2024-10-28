import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "j-min/reco_sd14_coco",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")