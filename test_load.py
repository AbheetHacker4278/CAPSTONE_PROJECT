import tensorflow as tf
import os

WEIGHTS_PATH = 'weights/modeldense1.h5'

# Redirect stdout to file
import sys
sys.stdout = open('load_test_result.txt', 'w')
sys.stderr = sys.stdout

print(f"Testing load_model on {WEIGHTS_PATH}")
try:
    model = tf.keras.models.load_model(WEIGHTS_PATH)
    print("SUCCESS: load_model worked!")
    model.summary()
except Exception as e:
    print(f"FAILED: load_model error: {e}")

print("-" * 20)
print("Testing load_weights on created model")

try:
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization
    from tensorflow.keras.regularizers import l1_l2
    
    conv_base = DenseNet201(input_shape=(224, 224, 3), include_top=False, pooling='max', weights='imagenet')
    model = Sequential()
    model.add(conv_base)
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu', kernel_regularizer=l1_l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='softmax'))
    
    model.load_weights(WEIGHTS_PATH)
    print("SUCCESS: load_weights worked!")
except Exception as e:
    print(f"FAILED: load_weights error: {e}")
