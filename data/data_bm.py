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
import numpy as np
import pickle
import time
import math
from collections import deque

def resize_and_crop(image, bucket_res):
    
    original_width, original_height = image.size
    bucket_width, bucket_height=bucket_res
    # 计算图像的宽高比和桶的宽高比
    image_aspect = original_width / original_height
    bucket_aspect = bucket_width / bucket_height
    
    # 缩放图像
    if image_aspect > bucket_aspect:
        # 如果图像宽高比大于桶宽高比，调整宽度
        new_height = bucket_height
        new_width = int(new_height * image_aspect)
    else:
        # 如果图像宽高比小于或等于桶宽高比，调整高度
        new_width = bucket_width
        new_height = int(new_width / image_aspect)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # 随机裁剪
    if new_width > bucket_width:
        left = new_width - bucket_width
        top = 0
        right = left + bucket_width
        bottom = bucket_height
    elif new_height > bucket_height:
        left = 0
        top = new_height - bucket_height
        right = bucket_width
        bottom = top + bucket_height
    else:
        left = 0
        top = 0
        right = new_width
        bottom = new_height
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # 保存处理后的图像
    return cropped_image

def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    def __init__(self, bucket_file, valid_ids=None, max_size=(1536,1536), divisible=32, step_size=8, min_dim=768, base_res=(1024,1024), bsz=1, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=1216, debug=False):
        with open(bucket_file, "rb") as fh:
            self.res_map = pickle.load(fh)
        if valid_ids is not None:
            new_res_map = {}
            valid_ids = set(valid_ids)
            for k, v in self.res_map.items():
                if k in valid_ids:
                    new_res_map[k] = v
            self.res_map = new_res_map
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)
        self.epoch_prng = get_prng(epoch_seed) # separate prng for sharding use for increased thread resilience
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()
        self.assign_buckets()
        self.start_epoch()

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def assign_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        self.buckets = {}
        self.aspect_errors = []
        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(self.aspects - aspect).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                if self.debug:
                    self.aspect_errors.append(error)
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]
        if self.debug:
            timer = time.perf_counter() - timer
            self.aspect_errors = np.array(self.aspect_errors)
            print(f"skipped images: {skipped}")
            print(f"aspect error: mean {self.aspect_errors.mean()}, median {np.median(self.aspect_errors)}, max {self.aspect_errors.max()}")
            for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
                print(f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, entries {len(self.buckets[bucket_id])}")
            print(f"assign_buckets: {timer:.5f}s")

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            timer = time.perf_counter()
        if world_size is not None:
            self.world_size = world_size
        if global_rank is not None:
            self.global_rank = global_rank

        # select ids for this epoch/rank
        index = np.array(sorted(list(self.res_map.keys())))
        index_len = index.shape[0]
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.bsz * self.world_size))]
        #print("perm", self.global_rank, index[0:16])
        index = index[self.global_rank::self.world_size]
        self.batch_total = index.shape[0] // self.bsz
        assert(index.shape[0] % self.bsz == 0)
        index = set(index)

        self.epoch = {}
        self.left_over = []
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = np.array([post_id for post_id in self.buckets[bucket_id] if post_id in index])
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    self.left_over.extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]
                if len(self.epoch[bucket_id]) == 0:
                    del self.epoch[bucket_id]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        # check if no data left or no epoch initialized
        if self.epoch is None or self.left_over is None or (len(self.left_over) == 0 and not bool(self.epoch)) or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            if len(self.left_over) >= self.bsz:
                bucket_probs = [len(self.left_over)] + [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
                bucket_ids = [-1] + bucket_ids
            else:
                bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_lens = bucket_probs
            bucket_probs = bucket_probs / bucket_probs.sum()
            bucket_ids = np.array(bucket_ids, dtype=np.int64)
            if bool(self.epoch):
                chosen_id = int(self.prng.choice(bucket_ids, 1, p=bucket_probs)[0])
            else:
                chosen_id = -1

            if chosen_id == -1:
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                self.prng.shuffle(self.left_over)
                batch_data = self.left_over[:self.bsz]
                self.left_over = self.left_over[self.bsz:]
                found_batch = True
            else:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    # return bucket batch and resolution
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.left_over.extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]

            assert(found_batch or len(self.left_over) >= self.bsz or bool(self.epoch))

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"bucket probs: " + ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs*100))))
            print(f"chosen id: {chosen_id}")
            print(f"batch data: {batch_data}")
            print(f"resolution: {resolution}")
            print(f"get_batch: {timer:.5f}s")

        self.batch_delivered += 1
        return (batch_data, resolution)

    def generator(self):
        if self.batch_delivered >= self.batch_total:
            self.start_epoch()
        while self.batch_delivered < self.batch_total:
            yield self.get_batch()

image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
def img2tensor(img,mask=None,device='cuda',size=(1024,1024)):
    # img = img.resize(size)
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
        kernel = np.ones((12,12), np.uint8)
        dilation_iterations = random.randint(1, 3)
        mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
        mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
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


class dataset_laion(Dataset):
    def __init__(self, path='/x22201004/data_laion',pkl="/x22201004/resolutions.pkl",conds=['openpose','sketch','lineart','depth','scribble'],bs=4 ,device='cuda',drop_cond_rate=0.3):
        super(dataset_laion, self).__init__()
        self.imgpath=os.path.join(path,'image')
        self.text_path=os.path.join(path,'caption1')
        conds_path=[os.path.join(path,'conditions',cond) for cond in conds]
        self.conds_path=dict(zip(conds,conds_path))
        self.bm = BucketManager(pkl, debug=False, bsz=bs, world_size=1, global_rank=0)
        self.conds=conds
        self.drop_cond_rate=drop_cond_rate
        self.device=device
    def __getitem__(self, idx):
        batch_imgname,res=self.bm.get_batch()

        imgs=[]
        captions=[]
        cond_imgs=dict(zip(self.conds,[[],[],[],[],[]]))
        keep_conds=dict(zip(self.conds,[[],[],[],[],[]]))
        for i , item in enumerate(batch_imgname):
            img=Image.open(os.path.join(self.imgpath,batch_imgname[i]))
            mask_path='/x22201004/data_laion/instants'
            if os.path.isfile(os.path.join(mask_path,batch_imgname[i].split('.')[0]+'.npz')):
                data = np.load(os.path.join(mask_path,batch_imgname[i].split('.')[0]+'.npz'))
                masks,cls=data['masks'],data['cls']
                b,_,_=masks.shape
                if b==1:
                    masks=np.concatenate((masks,1-masks), axis=0)
                    b*=2
                mask_index=deque(random.sample(range(b), b))


            else:
                masks=None
            size=img.size
            imgs.append(img2tensor(resize_and_crop(img,res),device=self.device)*2-1)
            if random.random()>0.5:
                with open(os.path.join(self.text_path,batch_imgname[i].split('.')[0]+'.txt'), "r") as f:  
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
                    captions.append(caption)
            else:
                with open(os.path.join('/x22201004/data_laion/caption',batch_imgname[i].split('.')[0]+'.txt'), "r") as f:  
                    data = f.read() 
                    caption=data


                
                captions.append(caption)
            
            for cond in self.conds_path.keys():
                
                cond_img_path=os.path.join(self.conds_path[cond],item)

                if cond=='openpose':

                    if os.path.isfile(cond_img_path):
                        img_cond=Image.open(cond_img_path)
                        img_cond=img_cond.resize(size, Image.LANCZOS)
                        keep_cond=0
                        if masks is not None and len(mask_index)>0:
                            mask=masks[mask_index.popleft()][:,:,None]
                        else :
                            if masks is not None:
                                mask=np.zeros_like(masks[0])[:,:,None]
                            else: mask=None

                    else:
                        img_cond=Image.new("RGB", res, (0, 0, 0))
                        keep_cond=1
                        mask=None
                else :
                    if random.random()>self.drop_cond_rate and os.path.isfile(cond_img_path):
                        img_cond=Image.open(cond_img_path)
                        img_cond=img_cond.resize(size, Image.LANCZOS)
                        keep_cond=0
                        if masks is not None and len(mask_index)>0:
                            mask=masks[mask_index.popleft()][:,:,None]
                        else:
                            if masks is not None:
                                mask=np.zeros_like(masks[0])[:,:,None]
                            else:mask=None
                    else:
                        img_cond=Image.new("RGB", res, (0, 0, 0))
                        keep_cond=1
                        mask=None
                keep_conds[cond].append(keep_cond)
                cond_imgs[cond].append(img2tensor(resize_and_crop(img_cond,res),mask=mask,device=self.device))
        for cond in self.conds_path.keys():
                keep_conds[cond]=torch.tensor(keep_conds[cond],device=self.device)
                
                cond_imgs[cond]=torch.stack(cond_imgs[cond])
        imgs=torch.stack(imgs)
    
        return{"pixel_values": imgs, 'conds': cond_imgs, 'keep_cond':keep_conds,'caption':captions}

    def __len__(self):
        return self.bm.batch_total
    

    
def my_collate_fn(batch):
    data=batch[0]
    return data
    






if __name__ == "__main__":
    dataset=dataset_laion(drop_cond_rate=0)


    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=my_collate_fn
    )
    i=0
    for data in train_dataloader:
        # print(data["pixel_values"].shape)
        # print(data['conds']['sketch'].shape)
        # print(data['keep_cond'])
        print(len(data['caption']))
        # print(data['conds']['scribble'])
        i+=1
        print(i)
    #     if i >10:
    #         break
    # print(i)
    #     vae = AutoencoderKL.from_pretrained('/x22201004/sdxl_weights/sdxl1.0/vae',torch_dtype=torch.float16)
    # pipe = StableDiffusionXLAdapterPipeline.from_pretrained("/x22201004/sdxl_weights/sdxl1.0", vae=vae,  torch_dtype=torch.float16, variant="fp16",).to("cuda")
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # pipe.to('cuda')
    # print(pipe)
