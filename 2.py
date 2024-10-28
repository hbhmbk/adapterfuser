# import os
# from tqdm import tqdm
# from PIL import Image
# from controlnet_aux import MidasDetector,PidiNetDetector,LineartDetector,OpenposeDetector
# conds=['depth', 'sketch', 'lineart', 'openpose']
# detectors=[MidasDetector.from_pretrained('/x22201004/controlnet_aux').to('cuda'),PidiNetDetector.from_pretrained('/x22201004/controlnet_aux').to('cuda'),LineartDetector.from_pretrained('/x22201004/controlnet_aux').to('cuda'),OpenposeDetector.from_pretrained('/x22201004/controlnet_aux').to('cuda')]

# imgs=os.listdir('/x22201004/coco_val/image')

# img_path='/x22201004/coco_val/image'
# output_path='/x22201004/coco_val/conditions'
# for i in tqdm(range(len(imgs))):
#     img_name=imgs[i]
#     img=Image.open(os.path.join(img_path,img_name))
#     for j in range(len(conds)):
#         cond=conds[j]
#         detector=detectors[j]
#         out=detector(img,detect_resolution=512, image_resolution=1024)
#         size=out.size
#         out.save(os.path.join(output_path,cond,img_name))
#     img=img.resize(size)
#     img.save(os.path.join('/x22201004/data/image1',img_name))

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    T2IAdapter,
    UNet2DConditionModel,
)
from pipeline.my_pipeline import   StableDiffusionXLAdapterPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
# from model.adapter import Adapter_XL
from data.data_bm import dataset_laion,my_collate_fn
from data.data import dataset_laion_
from model .utils import get_adapter
from model.my_adapter import Adapter_XL
from safetensors.torch import load_file
from data.data import dataset_laion_
weight_dtype=torch.float16


state_dict = load_file("/x22201004/checkpoint-148000/model.safetensors")
t2iadapter_fuser=Adapter_XL().to('cuda')

# 加载权重到模型
t2iadapter_fuser.load_state_dict(state_dict)
adapters=get_adapter(dtype=weight_dtype)
# vae1 = AutoencoderKL.from_pretrained('/x22201004/sdxl_weights/sdxl1.0/vae',torch_dtype=weight_dtype)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                                '/x22201004/sdxl_weights/sdxl1.0',
                                   adapter=None,
                                    torch_dtype=weight_dtype,
                                    
                                )


pipe.adapters=adapters
pipe.adapter_fuser=t2iadapter_fuser

pipe.to('cuda')


test_dataset=dataset_laion_(path='/x22201004/testimg',drop_cond_rate=0.0,drop_caption_rate=0)
                    # test_dataloader = torch.utils.data.DataLoader(
                    #         test_dataset,
                    #         batch_size=1,
                    #         shuffle=True,
                            
                    #         )
test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        shuffle=True,
                        batch_size=1,
                        num_workers=0
                    )
i=0
for data in test_dataloader:
    keep_cond=data['keep_cond']
    prompt = data['caption']
    negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
    gen_images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=data['conds'],
    num_inference_steps=25,
    adapter_conditioning_scale=1,
    guidance_scale=7.5,
    keep_cond=keep_cond
    
    ).images[0]
    print('文本提示长度：{}'.format(len(prompt[0].split())))
    gen_images.save('/x22201004/outputs/{}.jpg'.format(i))
    print(prompt[0])
    # output_path=os.path.join('/x22201004/outputs_fuser/caption','{}.txt'.format(i))
    # with open(output_path, 'w') as file:
    #     file.write(data['caption'][0])
    # for name in data['conds'].keys():
    #     img=data['conds'][name][0]
    #     img*=255
    #     img=img.permute(1,2,0).cpu().numpy() 
    #     img= img.astype(np.uint8)
    #     img = Image.fromarray(img)
    #     img.save('/x22201004/coco_val/conditions/outputs/conds/{}_{}'.format(i,name)+'.jpg')
    #     image=data["pixel_values"][0]
    #     image=(image+1)*127.5
    #     img=image.permute(1,2,0).cpu().numpy()
    #     img= img.astype(np.uint8)
    #     img = Image.fromarray(img)
    #     img.save('/x22201004/coco_val/conditions/outputs/image/{}'.format(i)+'.jpg')
    i+=1  