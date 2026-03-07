import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths to dataset
DATASET_PATH = 'external_repo/mammography_images/mammography_images/'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
TRAIN_CSV = os.path.join(DATASET_PATH, 'Training_set.csv')

def custom_preprocess(img):
    """Applies the same preprocessing used in model_utils.py"""
    # img is delivered as float32 RGB by ImageDataGenerator if target_size is set
    # but we need uint8 for OpenCV operations
    img = img.astype('uint8')
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 2. Sharpening
    kernel = np.array([[0,-0.5,0], [-0.5,3,-0.5], [0,-0.5,0]])
    img = cv2.filter2D(img, -1, kernel)
    
    # 3. DenseNet specific normalization
    img = img.astype('float32')
    img = densenet_preprocess(img)
    
    return img

def train_image_model():
    print("Preparing data generators...")
    df = pd.read_csv(TRAIN_CSV)
    
    # Stratified split for validation
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)

    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocess,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=custom_preprocess)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
    )

    print("Building model...")
    conv_base = DenseNet201(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='max')
    
    model = Sequential()
    model.add(conv_base)
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu', kernel_regularizer=l1_l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='softmax'))

    # Fine-tuning: Freeze most layers, train top and some deeper layers
    # As per original repo: train_layers = [layer for layer in conv_base.layers[::-1][:5]]
    # We'll stick to a similar strategy but allow a bit more training if needed.
    for layer in conv_base.layers:
        layer.trainable = False
    
    # Unfreeze the last 20 layers of DenseNet
    for layer in conv_base.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_with_crossentropy' if tf.__version__ < '2.0' else 'categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    print("Starting training (limited to 5 epochs for sanity/demonstration, increase as needed)...")
    
    checkpoint = ModelCheckpoint('weights/modeldense1.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    os.makedirs('weights', exist_ok=True)

    history = model.fit(
        train_generator,
        epochs=5,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )

    print("Retraining of image model complete.")
    return history

if __name__ == "__main__":
    train_image_model()
