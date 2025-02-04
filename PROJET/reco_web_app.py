####### PARTIE 2 : INTERFACE WEB AVEC SYSTEME DE RECOMMENDATIONS #######

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pandas as pd
import requests
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import gradio as gr
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os

API_URL = os.getenv("API_URL", "http://genre_api:5000").strip('"')

# Chargement du modèle pré-entrainé
mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model = nn.Sequential(mobilenet.features, mobilenet.avgpool, torch.nn.Flatten())
model.eval()

df = pd.read_csv("URLlist.csv")  # Contient les PATH des affiches que l'on va recommender
df = pd.DataFrame(df)
df = df.rename(columns={"Unnamed: 0": "index"})

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fonction pour extraire les caractéristiques d'une image
def extract_features(image):
    try:
        # Assurez-vous que l'image est un objet PIL.Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            embeddings = model(image_tensor)
        return embeddings.squeeze().numpy()
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques : {e}")
        return None


def get_recommendations(image):
    print(f"Type d'image reçue : {type(image)}")
    print(f"Shape de l'image (si NumPy) : {image.shape if isinstance(image, np.ndarray) else 'N/A'}")

    # Extraire les caractéristiques de l'image reçue
    vector = extract_features(image)
    if vector is None:  # Si l'extraction a échoué
        raise ValueError("Erreur lors de l'extraction des caractéristiques.")

    # Envoyer à l'API Flask pour obtenir les recommandations
    response = requests.post(API_URL + "/reco", json={'vector': vector.tolist()})
    response.raise_for_status()
    similar_movies_indices = response.json()
    print(f"Réponse de l'API : {similar_movies_indices}")

    # Récupérer les PATH des images
    paths = df[df['index'].isin(similar_movies_indices)]['path'].tolist()
    print(f"Chemins des images similaires : {paths}")

    # Affichage des images similaires sur Gradio
    fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
    if len(paths) == 1:
        axs = [axs]
    for i, path in enumerate(paths):
        img = Image.open(path)
        axs[i].imshow(img)
        axs[i].axis('off')
    return fig
    

# Interface Gradio
if __name__ == "__main__":
    gr.Interface(
        fn=get_recommendations,
        inputs=gr.Image(type="numpy"),  
        outputs="plot",                
        live=True,
        description="Upload an image to get 5 movie recommendations.",  
    ).launch(server_port=8000, debug=True, share=True, server_name="0.0.0.0")
