import streamlit as st
import cv2
import os
from mtcnn import MTCNN
from PIL import Image
import numpy as np

def main():
    st.set_page_config(page_title="Multiple Images Training", page_icon="📸")
    st.title("Multiple Images Training")
    person_name = st.text_input("Enter the person's name:")
    num_images = st.number_input("Enter the number of images to be taken:", min_value=1, step=1)
    option = st.radio("Choose an option:", ("Upload Images", "Capture from Camera"))
    if option == "Upload Images":
        uploaded_files = st.file_uploader("Upload face images (JPG or PNG)", type=["jpg", "png"], accept_multiple_files=True)
        if uploaded_files is not None and len(uploaded_files) >= num_images:
            folder_path = os.path.join("database", person_name)
            if os.path.exists(folder_path):
                for i, uploaded_file in enumerate(uploaded_files[:num_images]):
                    image = Image.open(uploaded_file)
                    image.save(os.path.join(folder_path, f"{person_name}_{i+1}.jpg"))
                st.success(f"{num_images} images saved successfully.")
            else:
                st.warning(f"Folder '{person_name}' does not exist.")
    elif option == "Capture from Camera":
        if st.button("Start Camera"):
            if not person_name:
                st.warning("Please enter a valid name.")
                return
            folder_path = os.path.join("database", person_name)
            if os.path.exists(folder_path):
                detector = MTCNN()
                cap = cv2.VideoCapture(0)
                img_count = 0
                while img_count < num_images:
                    ret, frame = cap.read()
                    faces = detector.detect_faces(frame)
                    if faces:
                        x, y, w, h = faces[0]['box']
                        face_roi = frame[y:y+h, x:x+w]
                        img_count += 1
                        cv2.imwrite(os.path.join(folder_path, f"{person_name}_{img_count}.jpg"), face_roi)
                        st.success(f"Image {img_count} captured.")
                    st.image(frame, channels="BGR", use_column_width=True)
                cap.release()
            else:
                st.warning(f"Folder '{person_name}' does not exist.")
    if st.button("Delete Folder"):
        if not person_name:
            st.warning("Please enter a valid name.")
            return
        folder_path = os.path.join("database", person_name)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            os.rmdir(folder_path)
            st.success(f"Folder '{person_name}' deleted successfully.")
        else:
            st.warning(f"Folder '{person_name}' does not exist.")

if __name__ == "__main__":
    main()
