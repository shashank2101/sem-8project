import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from mtcnn import MTCNN

# Assuming IMAGE_SIZE and train_generator are defined in exp.py
# from exp import IMAGE_SIZE, train_generator

# Set constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
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
data_dir = 'D:/attendance/chrome/database1'

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Use subset for training data
)
model_path = 'trained_model_mobilenet.h5'

# Load the model
model = load_model(model_path)
# Initialize MTCNN detector
detector = MTCNN()
# Streamlit UI
st.title("Face Recognition")
# File uploader for image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
# Check if image is uploaded
if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Detect faces in the image
    faces = detector.detect_faces(img)

    # If faces are detected, process the first face found
    for i in faces:
        # Extract the first face found
        face_data = i
        x, y, w, h = face_data['box']
        
        # Add padding around the face to fit into 224x224 size
        padding = 30
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding
        
        # Ensure the bounding box does not go beyond the image boundaries
        x = max(x, 0)
        y = max(y, 0)
        
        # Crop the face region
        face_img = img[y:y+h, x:x+w]

        # Resize the face image to match the input size of your model
        face_img = cv2.resize(face_img, IMAGE_SIZE)

        # Convert the face image to arr ay and preprocess it for prediction
        face_array = image.img_to_array(face_img)
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
        face_array /= 255.  # Normalize pixel values

        # Perform prediction
        prediction = model.predict(face_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction[0])
        class_names = train_generator.class_indices  # Assuming train_generator is defined
        class_name = [k for k, v in class_names.items() if v == predicted_class_index][0]
        print('Predicted class:', class_name)
        # Display the predicted class
        st.write('Students detected:', class_name)
    else:
        st.write('No faces detected in the image.')
