import os
from tqdm import tqdm
from PIL import Image
from controlnet_aux import MidasDetector,PidiNetDetector,LineartDetector,OpenposeDetector
from controlnet_aux import HEDdetector
conds=['depth', 'sketch', 'lineart', 'openpose']

hed = HEDdetector.from_pretrained('/x22201004/controlnet_aux').to('cuda')
imgs=os.listdir('/x22201004/coco_val/image')

img_path='/x22201004/coco_val/image'
output_path='/x22201004/coco_val/conditions'
for i in tqdm(range(len(imgs))):
    img_name=imgs[i]
    img=Image.open(os.path.join(img_path,img_name))
    image = hed(img, detect_resolution=512, image_resolution=1024,scribble=True)
   
    image.save(os.path.join('/x22201004/data_laion/conditions/scribble',img_name))