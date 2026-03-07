import traceback
import sys
import os

# testing without the override

with open("debug_log.txt", "w") as log_file:
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        import model_utils
        log("Attempting load_cancer_model...")
        model = model_utils.load_cancer_model()
        if model:
            log("Success!")
        else:
            log("Returned None")
    except Exception:
        log(traceback.format_exc())

    log("\nDetailing create_model error if it fails:")
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import DenseNet201
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, BatchNormalization
        from tensorflow.keras.regularizers import l1_l2
        
        log("1. Creating DenseNet201...")
        conv_base = DenseNet201(input_shape=(224, 224, 3), include_top=False, pooling='max', weights='imagenet')
        log("2. Creating Sequential...")
        model = Sequential()
        log("3. Adding conv_base...")
        model.add(conv_base)
        log("4. Adding BatchNormalization...")
        model.add(BatchNormalization())
        log("5. Adding Dense...")
        model.add(Dense(2048, activation='relu', kernel_regularizer=l1_l2(0.01)))
        log("6. Adding BatchNormalization...")
        model.add(BatchNormalization())
        log("7. Adding softmax...")
        model.add(Dense(8, activation='softmax'))
        log("8. Freezing layers...")
        train_layers = [layer for layer in conv_base.layers[::-1][:5]]
        for layer in conv_base.layers:
            if layer not in train_layers:
                layer.trainable = False
        log("Finished successfully!")
    except Exception:
        log(traceback.format_exc())

