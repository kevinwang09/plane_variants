import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications.mobilenet_v2 as mn2

# load pre-trained model and data
model = mn2.MobileNetV2(weights='imagenet')

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prep_fn)

# All photos in all_info.csv
all_info = pd.read_csv("./output/all_info_2021.csv")
IMG_SIZE = (225, 400)

all_dataset = datagen.flow_from_dataframe(
    dataframe=all_info,
    directory=None,
    x_col="download_path",
    y_col="model",
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
    seed=1234)

with tf.device('/GPU:0'): 
    all_predictions = model.predict(all_dataset, verbose = 1)

all_predictions_decoded = mn2.decode_predictions(all_predictions, top=1)
all_predictions_class = [l[0][1] for l in all_predictions_decoded]
from collections import Counter
Counter([l for l in all_predictions_class])
all_info["qc_class"] = all_predictions_class
all_info.to_csv(path_or_buf="./output/all_info_filtered_for_planes_03May2022.csv")