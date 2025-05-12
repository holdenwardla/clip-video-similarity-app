import streamlit as st
import tempfile
import os
import torch
import clip
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames(video_path, every_n_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, image = cap.read()
    while success:
        if count % every_n_frames == 0:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            frames.append(pil_img)
        success, image = cap.read()
        count += 1
    cap.release()
    return frames

def get_clip_embeddings(frames):
    embeddings = []
    for img in frames:
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy()[0])
    return embeddings

# Streamlit UI
st.title("🎥 Video Similarity Detector (CLIP)")

video1 = st.file_uploader("Upload Video 1", type=["mp4", "mov", "avi"], key="v1")
video2 = st.file_uploader("Upload Video 2", type=["mp4", "mov", "avi"], key="v2")

if video1 and video2:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp1:
        temp1.write(video1.read())
        video1_path = temp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp2:
        temp2.write(video2.read())
        video2_path = temp2.name

    st.info("Extracting frames...")
    frames1 = extract_frames(video1_
