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

# Frame extractor
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

# Get CLIP embeddings
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
    frames1 = extract_frames(video1_path)
    frames2 = extract_frames(video2_path)

    if not frames1 or not frames2:
        st.error("Failed to extract frames.")
    else:
        st.info("Getting embeddings...")
        emb1 = get_clip_embeddings(frames1)
        emb2 = get_clip_embeddings(frames2)

        st.success("Calculating similarity...")
        similarity_matrix = cosine_similarity(np.array(emb1), np.array(emb2))
        avg_score = np.mean(similarity_matrix)
        st.metric("Average Similarity Score", f"{avg_score:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(similarity_matrix, cmap="viridis", ax=ax)
        ax.set_title("Frame-to-Frame Similarity")
        ax.set_xlabel("Video 2 Frames")
        ax.set_ylabel("Video 1 Frames")
        st.pyplot(fig)

        # Cleanup
        os.remove(video1_path)
        os.remove(video2_path)
