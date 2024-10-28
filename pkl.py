import os
import pickle
from PIL import Image
from tqdm import tqdm
# 图像数据集的路径
image_folder = "/x22201004/testimg1/image"

# 初始化字典以存储图像ID和分辨率
image_resolutions = {}

# 遍历文件夹中的所有图像文件
for filename in tqdm(os.listdir(image_folder)):
    if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):  # 你可以根据需要扩展文件类型
        image_path = os.path.join(image_folder, filename)
        with Image.open(image_path) as img:
            resolution = img.size  # 获取图像的分辨率
            image_resolutions[filename] = resolution  # 使用文件名作为图像ID

# 将字典保存到pickle文件
with open("pexels.pkl", "wb") as fh:
    pickle.dump(image_resolutions, fh)

print("testimg1.pkl 文件已创建。")