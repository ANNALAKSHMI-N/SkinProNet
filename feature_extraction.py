if True:
    from reset_random import reset_random

    reset_random()
import glob
import os
import cv2
import numpy as np
import tqdm
import cmapy
from utils import CLASSES
from keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from keras.models import Model
from keras.utils import load_img, img_to_array
from keras.utils.vis_utils import plot_model

SHAPE = (224, 224, 3)


def getEfficientNetV2S():
    print("[INFO] Building EfficientNetV2S Model")
    model = EfficientNetV2S(input_shape=SHAPE, include_top=False, weights="imagenet")
    plot_model(model, to_file="eff_model.png", show_shapes=True,show_layer_names=True)
    return model


def get_feature_map_model(model):
    layer_outputs = [layer.output for layer in model.layers[1:]]
    feature_map_model = Model(model.input, layer_outputs)
    plot_model(feature_map_model, to_file="featuremodel.png",show_layer_names=True)
    return feature_map_model


def get_image_to_predict(im_path):
    img = load_img(im_path, target_size=SHAPE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def get_feature(img, model):
    feature = model.predict(img, verbose=False)[0][0][0]
    feat = feature.flatten()
    return feat.tolist()


def get_feature_image(img, model):
    feature_map = model.predict(img, verbose=False)[5]
    feature_image = feature_map[0, :, :, -1]
    feature_image -= feature_image.mean()
    feature_image /= feature_image.std()
    feature_image *= 64
    feature_image += 128
    feature_image = np.clip(feature_image, 0, 255).astype("uint8")
    return feature_image


if __name__ == "__main__":
    DATA_DIR = "Data/preprocessed"
    SAVE_DIR = "Data/features"

    fe_model = getEfficientNetV2S()

    fm_model = get_feature_map_model(fe_model)

    features = []
    labels = []

    for cls in CLASSES:
        sd = os.path.join(SAVE_DIR, cls)
        os.makedirs(sd, exist_ok=True)
        images_list = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.JPG")))
        for img_path in tqdm.tqdm(
            images_list,
            desc="[INFO] Extracting EfficientNetV2S Features For Class :: {0}".format(
                cls
            ),
        ):
            im_name = os.path.basename(img_path)
            im_save_path = os.path.join(sd, im_name)
            im_to_pred = get_image_to_predict(img_path)
            im_fe = get_feature(im_to_pred, fe_model)
            features.append(im_fe)
            labels.append(CLASSES.index(cls))
            im_fm = get_feature_image(im_to_pred, fm_model)
            im_fm = cv2.resize(im_fm, SHAPE[:-1])
            im_fm = cv2.applyColorMap(im_fm, cmapy.cmap("viridis_r"))
            cv2.imwrite(im_save_path, im_fm)

    print("[INFO] Saving Features and Labels")
    f_path = os.path.join(SAVE_DIR, "features.npy")
    features = np.array(features, ndmin=2)
    np.save(f_path, features)
    labels = np.array(labels)
    l_path = os.path.join(SAVE_DIR, "labels.npy")
    np.save(l_path, labels)
