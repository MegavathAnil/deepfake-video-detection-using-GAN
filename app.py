import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import face_recognition
from torchvision import models
from torch import nn
from torch.utils.data.dataset import Dataset
import os

# Model Definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Preprocessing
def preprocess_frame(frame):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(frame)

class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_length=20):
        self.video_path = video_path
        self.sequence_length = sequence_length
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            faces = face_recognition.face_locations(frame)
            if faces:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            frames.append(preprocess_frame(frame))
        cap.release()
        return torch.stack(frames).unsqueeze(0)

# Prediction
def predict(model, video_path):
    dataset = VideoDataset(video_path)
    sm = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        frames = dataset[0]
        fmap, logits = model(frames)
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
    return "REAL" if prediction.item() == 1 else "FAKE", confidence

# Load Model
@st.cache_resource
def load_model():
    model = Model(2)
    model_path = os.path.join(os.path.dirname(__file__), "model", "df_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

# Streamlit UI
st.title("Deepfake Video Detection")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if os.path.exists(temp_path):
        cap = cv2.VideoCapture(temp_path)
        if cap.isOpened():
            st.write(f"Saved video at: {temp_path}")
            frame_placeholder = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB")
            cap.release()
        else:
            st.error("Error: Video file could not be opened.")
    else:
        st.error("Error saving video file.")
    
    model = load_model()
    result, confidence = predict(model, temp_path)
    
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}%")
    
