import glob
import os
import numpy as np
import tqdm
import cv2
from u2netpp import U2NetPP, IMG_SHAPE
from utils import CLASSES

IMG_DIM = IMG_SHAPE


def segment(path, mdl):
    img_ = cv2.imread(path, 0)
    img_ = cv2.resize(img_, IMG_DIM[:-1])
    img_ = np.expand_dims(img_, axis=-1)
    img_ = np.array([img_], ndmin=2).astype(np.float32)
    pred = (mdl.predict(img_, verbose=False) > 0.2) * 1
    pred = pred[0]
    pred[pred > 0] = 255
    return pred.reshape(pred.shape[:-1])


if __name__ == "__main__":
    data_dir = "Data/data"
    save_dir = "Data/segmented"
    model = U2NetPP()
    model.load_weights("u2netpp/model.h5")
    for cls in CLASSES:
        images_list = sorted(glob.glob(os.path.join(data_dir, cls, "*.jpg")))
        sd = os.path.join(save_dir, cls)
        os.makedirs(sd, exist_ok=True)
        for img_path in tqdm.tqdm(
            images_list, desc="[INFO] Segmenting Class :: {0}".format(cls)
        ):
            im_save_path = os.path.join(sd, os.path.basename(img_path))
            if not os.path.isfile(im_save_path):
                if cls != "NORM":
                    mask = segment(img_path, model)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, IMG_DIM[:-1])
                    segmented = cv2.bitwise_and(img, img, mask=mask.astype(np.int8))
                    cv2.imwrite(im_save_path, segmented)
