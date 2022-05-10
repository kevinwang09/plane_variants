# Importing libraries
from tabnanny import verbose
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import collections
import json

# Change this when running a different dated version
date = "06May2022"
model_name = "./model/all9variants_size400_" + date
output_csv_name = "./output/all9classes_predictions_" + date + ".csv"
output_json_name = "./output/all9classes_index_" + date + ".json"

# Setting up the session for tensorflow
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=8,
    inter_op_parallelism_threads=0)
sess = tf.compat.v1.Session(config=session_conf)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

# Loading all_info CSV sheet
all_info = pd.read_csv("./output/all_info_size400_stage0filtered_" + date + ".csv")
all_info = all_info.loc[all_info.qc_class.isin(["airliner", "warplane", "airship"]), ].reset_index()
all_info.sample(10)

# Randomly split the data into traing/test/validation sets
# all_info.loc[:, "download_path"] = all_info.download_path.str.replace("\\./", "/content/drive/MyDrive/GitHub_Projects/plane_variants/")
all_info_rows = all_info.shape[0]
np.random.seed(10)
all_info["purpose"] = np.random.choice(
    ["train", "test", "validation"], size=all_info_rows, replace=True, p=[0.64, 0.16, 0.2])
# all_info["classes"] = np.where(
#     all_info.model.isin(["B737"]),
#     all_info["model"] + "-" + all_info["variant"].astype(str).str[0],
#     all_info.model)
# ## Editing B737 variants
# all_info["classes"] = np.where(
#     all_info.classes.isin(["B737-2", "B737-3","B737-4","B737-5"]),
#     "B737-g2", 
#     np.where(
#         all_info.classes.isin(["B737-6","B737-7","B737-8"]),
#         "B737-g3",
#         np.where(
#             all_info.classes.isin(["B737-9"]),
#             "B737-g4",
#             all_info.classes
#         )))
all_info["classes"] = all_info["model"]
all_info.classes.value_counts()
all_info = all_info.groupby("classes").filter(lambda x: len(x) > 50)

train_df = all_info.loc[all_info.purpose == "train"]
test_df = all_info.loc[all_info.purpose == "test"]
validation_df = all_info.loc[all_info.purpose == "validation"]
print(train_df.download_path.head())

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

# Creating train/test/validation dataset from dataframe
datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prep_fn)
IMG_SIZE = (225, 400)

train_dataset = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="download_path",
    y_col="classes",
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=1234)

test_dataset = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col="download_path",
    y_col="classes",
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=1234)

validation_dataset = datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=None,
    x_col="download_path",
    y_col="classes",
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=1234)

print(collections.Counter(train_dataset.labels))
print(train_dataset.class_indices)

train_dataset[0][0].shape

# plt.imshow(((train_dataset[0][0][0, :, :, :])+1)/2)
# plt.show()

print(np.max(train_dataset[0][0][0, :, :, :]))
print(np.min(train_dataset[0][0][0, :, :, :]))

# Setting up base model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Setting up the prediction layer as a Dense layer
prediction_layer = tf.keras.layers.Dense(int(max(train_dataset.classes) + 1))
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2)
])

inputs = tf.keras.Input(shape=(225, 400, 3))
x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    #   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
model.summary()

initial_epochs = 3

loss0, accuracy0 = model.evaluate(validation_dataset, verbose=1)

with tf.device('/GPU:0'): 
    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()), 1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0, 1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
    metrics=['accuracy'])
model.summary()
len(model.trainable_variables)

fine_tune_epochs = 100
total_epochs = initial_epochs + fine_tune_epochs

callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    min_delta=0.01,
    verbose=1, 
    mode="max",
    restore_best_weights=True)

with tf.device('/GPU:0'): 
    history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
                         callbacks=[callback])

model.save(model_name)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 2.0])
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

model = keras.models.load_model(model_name)

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

all_info_output = all_info.copy()
all_predictions = model.predict(all_dataset, verbose=1)
all_indices = all_dataset.class_indices
all_indices_flipped = dict(zip(all_indices.values(), all_indices.keys()))

plane_model_names = ["pred_" + m for m in list(all_indices.keys())]
all_predictions_df = pd.DataFrame(all_predictions, columns=plane_model_names)
all_predictions_class_indices = list(np.apply_along_axis(func1d=np.argmax, arr=all_predictions, axis=1))
all_predictions_class = [all_indices_flipped[k] for k in all_predictions_class_indices]
all_predictions_df["prediction_class"] = all_predictions_class

all_info_output = pd.concat([all_info, all_predictions_df], axis=1)
all_info_output.loc[:,all_info_output.columns[2:]]
all_info_output.to_csv(path_or_buf=output_csv_name, index=False)

with open(output_json_name, "w") as outfile:
    json.dump(all_indices_flipped, outfile)
