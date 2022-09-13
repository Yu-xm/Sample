import pandas as pd
import os
import numpy as np
import shutil

df = pd.read_csv('image_info.csv')


image_id_ls= list(np.asarray(df['id']))


path_images = '/Users/mryu/Desktop/科研/data/Fakeddit/data/images'
image_ls_501 = os.listdir(path_images)
# print(image_ls)

count = 0
for image in image_ls_501:
    image_name = image.replace('.jpg', '')
    if image_name not in image_id_ls:
        count = count + 1
        print(image_name)

print(count)

print("done")
