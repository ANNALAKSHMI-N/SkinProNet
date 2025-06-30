import glob
import os
import time
import numpy as np
import tqdm
import cv2

from utils import CLASSES


def wiener_filter(img):
    dft = np.fft.fft2(img)
    pspec = (np.abs(dft)) ** 2
    noise = 5
    wiener = pspec / (pspec + noise)
    wiener = wiener * dft
    restored = np.fft.ifft2(wiener)
    restored = np.real(restored)
    restored = restored.clip(0, 255).astype(np.uint8)
    return restored


def contrast_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    lab_planes[0] = clahe_.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(path):
    img = cv2.imread(path)
    wf = wiener_filter(img)
    ce = contrast_enhance(wf)
    return {"Wiener Filtered": wf, "Contrast Enhanced": ce}


if __name__ == "__main__":
    data_dir = "Data/data"
    save_dir = "Data/preprocessed"
    for cls in CLASSES:
        images_list = sorted(glob.glob(os.path.join(data_dir, cls, "*.jpg")))
        save_path = os.path.join(save_dir, cls)
        os.makedirs(save_path, exist_ok=True)
        for img_path in tqdm.tqdm(
            images_list, desc="[INFO] PreProcessing Class :: {0}".format(cls)
        ):
            im_save_path = os.path.join(save_path, os.path.basename(img_path))
            if not os.path.isfile(im_save_path):
                pp = preprocess(img_path)
                cv2.imwrite(im_save_path, pp["Contrast Enhanced"])
    time.sleep(0.2)
