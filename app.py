import streamlit as st
import cv2
import numpy as np
import torch
import os
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_objects(image):
    results = model(image)
    detection_image = np.array(results.render()[0])
    return detection_image

def main():
    st.title("Real-time Object Detection with YOLOv5")
    st.sidebar.header("Options")
    
    option = st.sidebar.selectbox("Choose an option", ["Upload Image", "Use Webcam", "Upload Video"])
    
    if option == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image = cv2.cvtColor(cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1), cv2.COLOR_BGR2RGB)
            detection_image = detect_objects(image)
            st.image(detection_image, channels="RGB")
    
    elif option == "Use Webcam":
        cap = cv2.VideoCapture(0)
        stop_button = st.button("Stop", key="stop_button")  # Add unique key to the button
        detection_placeholder = st.empty()  # Placeholder for detection image
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pil_image =Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_image = detect_objects(pil_image)
            pil_image = Image.fromarray(detection_image)
            detection_placeholder.image(pil_image, channels="RGB", use_column_width=True)
            if stop_button:
                break
        cap.release()
    
    elif option == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4"])
        if uploaded_video is not None:
            video_path = 'temp_video.mp4'
            with open(video_path, 'wb') as f:
                f.write(uploaded_video.read())
            cap = cv2.VideoCapture(video_path)
            stop_button_key = "stop_video_button_{}".format(id(cap))  # Generate a unique key
            stop_button = None
            stframe = st.empty()  # Placeholder for displaying video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                detection_image = detect_objects(frame)
                stframe.image(detection_image, channels="RGB", use_column_width=True)
                if stop_button is None:
                    stop_button = st.button("Stop Video", key=stop_button_key)  # Create the button only once
                if stop_button:
                    break
            cap.release()
            st.button("Remove Video", on_click=lambda: os.remove(video_path))



if __name__ == "__main__":
    main()
