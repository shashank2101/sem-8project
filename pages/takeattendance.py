import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from mtcnn import MTCNN
import streamlit as st
import os
import face_recognition


IMAGE_SIZE=(224,224)
BATCH_SIZE=32
detector = MTCNN()
model = load_model('trained_model_mobilenet.h5')

# Function to extract face encodings from an image
def extract_face_encodings(image_path):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    return face_encodings

# Function to compare input image with face encodings in the output directory
def find_matching_image(input_image_path, output_directory):
    input_face_encodings = extract_face_encodings(input_image_path)
    if not input_face_encodings:
        return "No face detected in the input image."
    
    for root, dirs, files in os.walk(output_directory):
        for file in files:
            if file.endswith('.npy'):
                output_image_path = os.path.join(root, file)
                output_face_encodings = np.load(output_image_path)
                for input_face_encoding in input_face_encodings:
                    results = face_recognition.compare_faces([output_face_encodings], input_face_encoding,tolerance=0.55)
                    if results[0]:
                        return True  # Return the matching image file name

    return False

# Example usage
# input_image_path = 'D:/attendance/chrome/database1/ranbir/r10.jpg'
output_directory = 'database2'

# @st.cache_data
# def long_running_function(param1, param2):
#     return 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Split 20% of data for validation
    )
data_dir = 'D:/attendance/chrome/database1'

    # Generate data batches from directory
train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'  # Use subset for training data
    )
# @st.cache(allow_output_mutation=True)
def predict_from_image(img):
    
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
        target_size = (224, 224)
        face_img = cv2.resize(face_img, target_size)

        # Convert the face image to array and preprocess it for prediction
        face_array = image.img_to_array(face_img)
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
        face_array /= 255.  # Normalize pixel values
        
        # Perform prediction
        prediction = model.predict(face_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction[0])

        # Map class index to class name
        class_names = train_generator.class_indices  # Assuming train_generator is defined
        class_name = [k for k, v in class_names.items() if v == predicted_class_index][0]

        return class_name
    else:
        return 'No faces detected in the image.'

# Streamlit UI
st.title('Face Recognition')
option = st.sidebar.selectbox('Select Input Source', ['Upload Image', 'Camera'])

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the  to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if find_matching_image(uploaded_file):
        # Predict from the image
            prediction = predict_from_image(img)
            st.write('Predicted class:', prediction)
elif option == 'Camera':
    st.write('Opening camera...')
    
    # Open camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Unable to capture frame")
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        st.image(rgb_frame, channels='RGB', use_column_width=True)
        if find_matching_image(frame):
        # Predict from the image
            prediction = predict_from_image(frame)
        st.write('Predicted class:', prediction)
        
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
    
    st.write('Camera option is not implemented yet.')
