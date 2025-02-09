import gradio as gr
from PIL import Image
import requests
import io
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

# Configuration de l'API
API_URL = os.getenv("API_URL", "http://localhost:5000").strip('"')

# ---------------------- Partie 1 : Prédiction du genre ----------------------

def recognize_genre(image):
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    response = requests.post(API_URL + "/predict", files={"file": img_binary.getvalue()})
   
    if response.status_code == 200:
        prediction = response.json().get("predicted_genre", "Error: No prediction")
        return prediction  
    return f"Error: {response.status_code}"

# ---------------------- Partie 2 : Recommandations de films ----------------------

mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model = nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten())
model.eval()

df = pd.read_csv("URLlist.csv")  
df = pd.DataFrame(df).rename(columns={"Unnamed: 0": "index"})

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embeddings = model(image_tensor)
    return embeddings.squeeze().numpy()

def get_recommendations(image):
    vector = extract_features(image)
    if vector is None:
        raise ValueError("Erreur lors de l'extraction des caractéristiques.")
   
    response = requests.post(API_URL + "/reco", json={'vector': vector.tolist()})
    response.raise_for_status()
    similar_movies_indices = response.json()
   
    paths = df[df['index'].isin(similar_movies_indices)]['path'].tolist()

    fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
    if len(paths) == 1:
        axs = [axs]
    for i, path in enumerate(paths):
        img = Image.open(path)
        axs[i].imshow(img)
        axs[i].axis('off')
    return fig

# ---------------------- Partie 3 : Détection d'anomalies ----------------------

def is_anomaly(image):
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    response = requests.post(API_URL + "/anomaly", files={"file": img_binary.getvalue()})
   
    if response.status_code == 200:
        prediction = response.json().get("is_a_poster", "Error: No prediction")
        return prediction  
    return f"Error: {response.status_code}"

# ---------------------- Interface Gradio ----------------------

with gr.Blocks(css="""
    body, html { background-color: #121212 !important; color: white !important; font-family: Arial, sans-serif; }
    h1, h2, h3, p, label { color: white !important; }
    .gradio-container { max-width: 800px; margin: auto; background-color: #121212 !important; padding: 20px; border-radius: 10px; }
    .gr-image { border: 2px solid #FFD700; border-radius: 10px; }
    .gr-textbox { border: 2px solid #FFD700; border-radius: 10px; padding: 10px; font-weight: bold; color: white !important; }
    .gr-markdown { text-align: center; color: white !important; }
""") as demo:
    with gr.TabItem("Genre Prediction"):
        gr.Markdown("### Upload an image to predict its genre.")
        genre_input = gr.Image(type="numpy")
        genre_output = gr.Textbox()
        genre_input.change(fn=recognize_genre, inputs=genre_input, outputs=genre_output)

    with gr.TabItem("Movie Recommendations"):
        gr.Markdown("### Upload an image to get 5 movie recommendations.")
        reco_input = gr.Image(type="numpy")
        reco_output = gr.Plot()
        reco_input.change(fn=get_recommendations, inputs=reco_input, outputs=reco_output)

    with gr.TabItem("Anomaly Detection"):
        gr.Markdown("### Upload an image to predict if it is a movie poster or not.")
        anomaly_input = gr.Image(type="numpy")
        anomaly_output = gr.Textbox()
        anomaly_input.change(fn=is_anomaly, inputs=anomaly_input, outputs=anomaly_output)


if __name__ == "__main__":
    demo.launch(server_port=7860, debug=True, share=True, server_name="0.0.0.0")

	
