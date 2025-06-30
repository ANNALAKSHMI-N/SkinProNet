if True:
    from reset_random import reset_random

    reset_random()

import glob
import os
import shutil

import cv2
import numpy as np
import tqdm
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array


if __name__ == "__main__":
    data_path = "Data/source/Normal"
    images_list = glob.glob(os.path.join(data_path, "*"))
    save_dir = "Data/data/NORM"
    os.makedirs(save_dir, exist_ok=True)
    for i, img_path in tqdm.tqdm(
        enumerate(images_list), desc="[INFO] Augmenting Normal Images :"
    ):
        aug = ImageDataGenerator(
            rotation_range=90,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        data = aug.flow(x, batch_size=1)
        shutil.copy(
            img_path,
            os.path.join(
                save_dir, "IMG_{0}_{1}.jpg".format(str(i).zfill(4), "0".zfill(4))
            ),
        )
        for j in range(15):
            it = data.next()
            image_ = it[0].astype("uint8")
            im_save_path = os.path.join(
                save_dir, "IMG_{0}_{1}.jpg".format(str(i).zfill(4), str(j + 1).zfill(4))
            )
            cv2.imwrite(im_save_path, cv2.cvtColor(image_, cv2.COLOR_RGB2BGR))
