import pandas as pd
import os
import numpy as np
import shutil
from tqdm import tqdm as tqdm


df_have_title = pd.read_csv('/Users/mryu/Desktop/images_all_processed/images_all_have_title/images_all_have_title.csv')
df_no_title = pd.read_csv('/Users/mryu/Desktop/images_all_processed/images_all_no_title/images_all_no_title.csv')

pbar = tqdm((len(df_have_title)+len(df_no_title)))

image_id_list_have = list(np.asarray(df_have_title['id']))
image_id_list_no = list(np.asarray(df_no_title['id']))
# print(image_id_list)

path_images = '/Users/mryu/Desktop/public_image_set'
image_ls = os.listdir(path_images)
# print(image_ls)

for image in image_ls:
    image_name = image.replace('.jpg', '')
    if image_name in image_id_list_have:
        shutil.move('/Users/mryu/Desktop/public_image_set/' + image,
                    '/Users/mryu/Desktop/images_all_processed/images_all_have_title/images')

    if image_name in image_id_list_no:
        shutil.move('/Users/mryu/Desktop/public_image_set/' + image,
                    '/Users/mryu/Desktop/images_all_processed/images_all_no_title/images')

    pbar.update(1)


print("done")
