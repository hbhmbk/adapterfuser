import numpy as np
import math
import random
from typing import Tuple
# from utils import make_bucket_resolutions
import torch
# from utils import setup_logging
# setup_logging()
# import logging
import imghdr
# logger = logging.getLogger(__name__)
import random
import json
import cv2
import os
import PIL
from PIL import   Image
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
import numpy as np
import torch
from einops import rearrange
import random
from diffusers.utils import PIL_INTERPOLATION, logging, replace_example_docstring, scale_lora_layers, unscale_lora_layers
from torch.utils.data import Dataset
# from pipeline.my_pipeline import   StableDiffusionXLAdapterPipeline
from diffusers import T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
def img2tensor(img,mask=None,device='cuda',size=(1024,1024)):
    img = img.resize(size)
    img = np.array(img, dtype=np.uint8)
    img = torch.from_numpy(img)
    if img.ndim == 2:
        img = img[:, :, None]
    assert img.ndim == 3
    h,w,c=img.shape
    if c !=3:
        img=torch.cat([img,img,img],dim=2)
    img = rearrange(img, 'h w c -> c h w')
    img=img.to(device)
    img=img/255
    if mask is not None:
        mask=np.concatenate([mask]*3,axis=2)
        kernel = np.ones((8,8), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        mask=torch.from_numpy(mask)
        mask = rearrange(mask, 'h w c -> c h w')
        img=img*mask.to(device)

    return img

def get_keep_conds(conds):
    keepcond={}
    for c in conds.keys():
        if conds[c].mean()==0 :
            keepcond[c]=torch.tensor(1,device='cuda')
        else:
            keepcond[c] = torch.tensor(0, device='cuda')
    return keepcond

def check_wordsinsentence(sentence):
    words_to_check = ["boy", "girl", "people", 'man', 'woman', 'person']
    for word in words_to_check:
        if word in sentence:
            return True
    return False


def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0

        image = image.transpose(2,0,1)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    return image


class dataset_laion_(Dataset):
    def __init__(self, path='/x22201004/data_laion',conds=['sketch','openpose','lineart','depth'], image_size=1024,device='cuda',drop_cond_rate=0.3,drop_caption_rate=0.4):
        super(dataset_laion_, self).__init__()
        self.all_name=os.listdir(os.path.join(path,'image'))
        
        self.conds=conds
        self.dataset_path=path
        self.device='cuda'
        self.drop_cond_rate=drop_cond_rate
        self.drop_caption_rate=drop_caption_rate
        self.img_size=image_size
    def __getitem__(self, idx):

        img_name=self.all_name[idx]
        img=Image.open(os.path.join(self.dataset_path,'image',img_name))
        # img=img.resize((1024,1024))
        # img=image_transforms(img).to(self.device)
        img=img2tensor(img,device=self.device)
        img=img*2-1
        if os.path.isfile(os.path.join(self.dataset_path,'caption1',img_name.replace('.jpg','.txt'))):
            with open(os.path.join(self.dataset_path,'caption1',img_name.replace('.jpg','.txt')), "r") as f:  
                data = f.read() 
                caption=data
                caption=caption.split()
                if caption[:2]==['The', 'image']:
                    caption=caption[3:]
        else:
            with open(os.path.join(self.dataset_path,'caption',img_name.replace('.jpg','.txt')), "r") as f:  
                data = f.read() 
                caption=data
                caption=caption.split()

        # if len(caption.split())<30:
        #     caption=caption_

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
            


        conds={}
        for c in self.conds:
            cond_img_path=os.path.join(self.dataset_path,'conditions',c,img_name)
            if c =='openpose':
                if os.path.isfile(cond_img_path):
                    cond_img=Image.open(cond_img_path)
                    conds[c]=img2tensor(cond_img)
                else :
                    conds[c]=torch.zeros_like(img)
            else:
                if random.random()>self.drop_cond_rate and os.path.isfile(cond_img_path):
                    cond_img=Image.open(cond_img_path)
                    conds[c]=img2tensor(cond_img)

                else:
                    conds[c]=torch.zeros_like(img)
        keep_cond=get_keep_conds(conds)

        return{"pixel_values": img, 'conds': conds, 'caption': caption,'keep_cond':keep_cond}
                





    def __len__(self):
        return len(self.all_name)
    

class dataset_laion_withmask(Dataset):
    def __init__(self, path='/x22201004/data_laion',conds=['sketch','openpose','lineart','depth'], image_size=1024,device='cuda',drop_cond_rate=0.2,drop_caption_rate=0.4):
        super(dataset_laion_withmask, self).__init__()
        self.all_name=os.listdir(os.path.join(path,'instance_mask'))
        self.conds=conds
        self.dataset_path=path
        self.device=device
        self.drop_cond_rate=drop_cond_rate
        self.drop_caption_rate=drop_caption_rate
        self.img_size=image_size
    def __getitem__(self, idx):

        name=self.all_name[idx]
        data = np.load(os.path.join(self.dataset_path,'instance_mask',name))
        masks,cls=data['mask'],data['cls']
        # kernel = np.ones((15, 15), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=1)
        b,h,w=masks.shape
        img_name=name.split('.')[0]+'.jpg'
        img=Image.open(os.path.join(self.dataset_path,'image',img_name))
        img=img2tensor(img,device=self.device)
        img=img*2-1
        
        with open(os.path.join(self.dataset_path,'caption',img_name.replace('.jpg','.txt')), "r") as f:  
            data = f.read() 
            caption=data

        mask_index=random.sample(range(b), b)
        conds={}
        i=0
        for c in self.conds:
            cond_img_path=os.path.join(self.dataset_path,'conditions',c,img_name)
            if c =='openpose':
                if os.path.isfile(cond_img_path):
                    cond_img=Image.open(cond_img_path)
                    if i<b:
                        mask=masks[mask_index[i]][:,:,None]
                        i+=1
                    else :
                        mask=np.zeros_like(mask[0])[:,:,None]
                    conds[c]=img2tensor(cond_img,mask)
                    
                else :
                    conds[c]=torch.zeros_like(img)
            else:
                if random.random()>self.drop_cond_rate and os.path.isfile(cond_img_path):
                    cond_img=Image.open(cond_img_path)
                    if i<b:
                        mask=masks[mask_index[i]][:,:,None]
                        i+=1
                    else :
                        mask=np.zeros_like(mask[0])[:,:,None]
                    conds[c]=img2tensor(cond_img,mask)
                    
                else:
                    conds[c]=torch.zeros_like(img)
        keep_cond=get_keep_conds(conds)

        return{"pixel_values": img, 'conds': conds, 'caption': caption,'keep_cond':keep_cond}
                





    def __len__(self):
        return len(self.all_name)






if __name__ == "__main__":
    dataset=dataset_laion(drop_cond_rate=0)
    # data=dataset[13]
    # c=data['conds']
    # image=data["pixel_values"]
    # image=(image+1)*127.5
    # img=image.permute(1,2,0).cpu().numpy()
    # img= img.astype(np.uint8)
    # # mask=mask*img

    # # 使用 PIL 创建图像
    # img = Image.fromarray(img)

    # img.save('/x22201004/ref_imgs/img.jpg')
    # for name in c.keys():
    #     img=c[name]
    #     img*=255
    #     img=img.permute(1,2,0).cpu().numpy()
    #     img= img.astype(np.uint8)
    # # mask=mask*img

    # # 使用 PIL 创建图像
    #     img = Image.fromarray(img)

    #     img.save('/x22201004/ref_imgs/{}'.format(name)+'.jpg')
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        

    )
    i=0
    for data in train_dataloader:
        print(data['conds'])
        i+=1
        if i >10:
            break
    #     vae = AutoencoderKL.from_pretrained('/x22201004/sdxl_weights/sdxl1.0/vae',torch_dtype=torch.float16)
    # pipe = StableDiffusionXLAdapterPipeline.from_pretrained("/x22201004/sdxl_weights/sdxl1.0", vae=vae,  torch_dtype=torch.float16, variant="fp16",).to("cuda")
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # pipe.to('cuda')
    # print(pipe)
