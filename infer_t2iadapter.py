from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL,MultiAdapter
from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
import torch
import numpy as np
from PIL import Image
from model .utils import get_adapter
import os
from PIL import Image
# adapter_sketch = T2IAdapter.from_pretrained(
#     "D:\sd\weights\\t2iadapter xl\\sketch", torch_dtype=torch.float16, varient="fp16"
# ).to("cuda")
# adapter_depth=T2IAdapter.from_pretrained(
#     "D:\sd\weights\\t2iadapter xl\\depth", torch_dtype=torch.float16, varient="fp16"
# ).to("cuda")
# adapter_line=T2IAdapter.from_pretrained(
#     "D:\sd\weights\\t2iadapter xl\\lineart", torch_dtype=torch.float16, varient="fp16"
# ).to("cuda")
weight_dtype=torch.float16
adapters=get_adapter(dtype=weight_dtype).adapters


negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    '/x22201004/sdxl_weights/sdxl1.0',  adapter=adapters['lineart'], torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')
cond_imgpath='/x22201004/coco_val/conditions/lineart'
caption_path='/x22201004/coco_val/caption1'
imgs=os.listdir(cond_imgpath)
j=0
for i in imgs:
    img=Image.open(os.path.join(cond_imgpath,i))
    name=i.split('.')[0]
    txt_path=os.path.join(caption_path,name+'.txt')
    if os.path.isfile(txt_path):
        with open(txt_path, "r") as f:  
            data = f.read() 
            caption=data
            caption=caption.split()
    if caption[:2]==['The', 'image']:
        caption=caption[3:]
    if len(caption)>63:
        words=caption[:64]
        index=0
        for i in reversed(range(len(words))):
            word=words[i]
            if word[-1]=='.'or word[-1]==',':
                index=i
                break
        words=words[:index+1]
        caption= ' '.join(words)

    else:
        caption= ' '.join(caption)
 
    gen_images = pipe(
        prompt=caption,
        negative_prompt=negative_prompt,
        image=img,
        num_inference_steps=25,
        adapter_conditioning_scale=1.0,
        guidance_scale=7.5,
        ).images[0]
    
    gen_images.save('/x22201004/out_t2iadapter/lineart'+'/{}.jpg'.format(j))
    j+=1


