import os
import tqdm
import cv2
import pandas as pd


DATA_PATH = "Data/source"
SAVE_DIR = "Data/data"


def prepare_lesion_images():
    df = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
    for row in tqdm.tqdm(df.values, desc="[INFO] Preparing Lesion Images :"):
        im_path = os.path.join(DATA_PATH, "Lesion", row[-2])
        diagnostic = row[17].upper()
        sd = os.path.join(SAVE_DIR, diagnostic)
        os.makedirs(sd, exist_ok=True)
        img = cv2.imread(im_path)
        cv2.imwrite(os.path.join(sd, row[-2].replace(".png", ".jpg")), img)


if __name__ == "__main__":
    prepare_lesion_images()
