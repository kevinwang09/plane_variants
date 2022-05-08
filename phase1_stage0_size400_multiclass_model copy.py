import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications.mobilenet_v2 as mn2
from collections import Counter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

date = "06May2022"
# load pre-trained model and data
model = mn2.MobileNetV2(weights='imagenet')

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prep_fn)

# All photos in all_info.csv
all_info = pd.read_csv("./img/size_400/all_info_"+date+".csv")
all_info.download_path = all_info.download_path.str.replace("./size_400", "./img/size_400")
# IMG_SIZE = (None, None)

all_dataset = datagen.flow_from_dataframe(
    dataframe=all_info,
    directory=None,
    x_col="download_path",
    y_col="model",
    # target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False,
    seed=1234)

with tf.device('/GPU:0'): 
    all_predictions = model.predict(all_dataset, verbose = 1)

all_predictions_decoded = mn2.decode_predictions(all_predictions, top=1)
all_predictions_class = [l[0][1] for l in all_predictions_decoded]
Counter([l for l in all_predictions_class])
all_info["qc_class"] = all_predictions_class
all_info.to_csv(path_or_buf="./output/all_info_size400_stage0filtered_" + date +".csv")