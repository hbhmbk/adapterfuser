# import torch

# from ultralytics import YOLO
# from PIL import Image
# # Load a model
# model = YOLO('/x22201004/yolov8x-seg.pt',
#              task='segment')  # load a pretrained model (recommended for training)
# import numpy as np
# import cv2
# import os
# from tqdm import tqdm
# # # Train the model
# # results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)

# # model.predict(source=r'D:\diffuser\new-yolov8\ultralytics-main\ultralytics\assets\\00000014.jpg', save=True,
# #               show=True)
# # results = model('/x22201004/data_laion/image/00000084.jpg')
# # masks=results[0].masks.data
# # b,h,w=masks.shape
# path='/x22201004/laion1/image'
# imgs=os.listdir(path)

# for i in tqdm(range(len(imgs))):
#     imgpath=os.path.join(path,imgs[i])
#     try:
#         with Image.open(imgpath).convert('RGB') as image:
#             image.verify()  # 仅验证文件是否损坏，不会加载图像
#     except (IOError, SyntaxError) as e:
#         continue
#     results = model(os.path.join(path,imgs[i]))
#     if results[0].masks is not None:
#         mask=results[0].masks.data
#         cls=results[0].boxes.cls
#         name=imgs[i]
#         name=name.split('.')[0]
#         b,h,w=mask.shape
#         if b >=2:
#             mask=mask.detach().cpu().numpy()
#             cls=cls.detach().cpu().numpy()
#             np.savez(os.path.join('/x22201004/laion1/instance_mask',name+'.npz'),mask=mask,cls=cls)
#             # torch.save(mask,os.path.join('/x22201004/data_laion/instance_mask',name+'.pt'))

        

# img=Image.open(path)
# img=img.resize((w,h))
# img = np.array(img, dtype=np.uint8)
# for i in range(b):
#     mask=masks[i][None,:,:]
#     mask=mask.permute(1,2,0)
#     mask=torch.cat([mask]*3,dim=2)

#     mask=mask.cpu().numpy()
#     kernel = np.ones((10, 10), np.uint8)

#     # 进行膨胀操作
#     mask = cv2.dilate(mask, kernel, iterations=1)
#     mask = mask.astype(np.uint8)
#     mask=mask*img

#     # 使用 PIL 创建图像
#     mask = Image.fromarray(mask)
#     mask.save('/x22201004/ref_imgs/{}.jpg'.format(i))


#     print(mask)

from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL,MultiAdapter
from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
import torch
import numpy as np
from PIL import Image
from model .utils import get_adapter
import os
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
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
controlnet = ControlNetModel.from_pretrained(
    "/x22201004/controlnet/lineart",
    torch_dtype=torch.float16
)

# when test with other base model, you need to change the vae also.
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    '/x22201004/sdxl_weights/sdxl1.0',
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
   
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')
cond_imgpath='/x22201004/coco_val/conditions/sketch'
caption_path='/x22201004/coco_val/caption1'
imgs=os.listdir(cond_imgpath)
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
    gen_images.save('/x22201004/out_controlnet/sketch'+'/{}.jpg'.format(name))