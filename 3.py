import os

path='/x22201004/data/caption1'
items=os.listdir(path)
for i in items:
    name=i
    with open(os.path.join(path,i), "r") as f:  
                data = f.read() 
                caption=data
                caption=caption.split()
                if caption[:2]==['The', 'image']:
                    caption=caption[3:]
                for i in range(len(caption)):
                       word=caption[i]
                       if word[-1]=='.':
                              index=i
                              break
                caption=caption[:index+1]
                caption= ' '.join(caption)
                print(caption)
    output_path=os.path.join('/x22201004/data/caption',name)
    with open(output_path, 'w') as file:
        file.write(caption)
# from ultralytics import YOLO
# import os
# import numpy as np
# from tqdm import tqdm
# model = YOLO("/x22201004/yolov8x-seg.pt")  # load an official model
# image_path='/x22201004/data/image1'
# imgs=os.listdir(image_path)
# for i in tqdm(range(len(imgs))):
#     img=imgs[i]
#     img_path=os.path.join(image_path,img)
#     outputdir=os.path.join('/x22201004/data/instants',img.split('.')[0])
#     # os.makedirs(os.path.join('/x22201004/data_laion/instants',outputdir), exist_ok=True)
#     results = model(img_path)
#     if results[0].masks ==None:
#         continue
#     masks=results[0].masks.data
#     cls=results[0].boxes.cls
#     cls=cls.detach().cpu().numpy()
#     masks=masks.detach().cpu().numpy()
#     masks = masks.astype(np.uint8)
#     np.savez_compressed(outputdir + '.npz', masks=masks,cls=cls)

# Predict with the model
# results = model("D:\模型结构\image\\000000082807.jpg")
# print(results)# predict on an image