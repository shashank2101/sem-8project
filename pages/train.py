import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import streamlit as st

# Set image size and batch size
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def train_model():
    # Define data directory
    data_dir = 'D:/attendance/chrome/database1'
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'  
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation' 
    )

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)  

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=0.001)  
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    st.write("Training started...")
    history = model.fit(train_generator,epochs=200,validation_data=validation_generator)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    st.pyplot()

    # Plot training & validation loss values
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    st.pyplot()

    # Identify the epoch with the highest validation accuracy or lowest validation loss
    optimal_epoch = np.argmax(history.history['val_accuracy']) + 1
    st.write("Optimal Epoch:", optimal_epoch)

    val_loss, val_accuracy = model.evaluate(validation_generator)
    st.write("Validation Loss:", val_loss)
    st.write("Validation Accuracy:", val_accuracy)

    model.save('trained_model_mobilenet.h5')

if st.button("Train Model"):
    train_model()
    st.write("Training completed")
