####### PARTIE 1 : INTERFACE WEB AVEC PREDICTION DU GENRE DU FILM #######

import gradio as gr
from PIL import Image
import requests
import io
import os

API_URL = os.getenv("API_URL", "http://genre_api:5000").strip('"')

# Fonction qui envoie l'image à l'API Flask et récupère la prédiction
def recognize_genre(image):
    
    image = Image.fromarray(image.astype('uint8'))
    
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    
    # Envoyer l'image en tant que données binaires à l'API Flask
    response = requests.post(API_URL + "/predict", files={"file": img_binary.getvalue()})
    
    # Si la requête est réussie, extraire la prédiction du genre
    if response.status_code == 200:
        prediction = response.json().get("predicted_genre", "Error: No prediction")
        return prediction  
    
    return f"Error: {response.status_code}"

# Interface Gradio
if __name__ == "__main__":
    gr.Interface(
        fn=recognize_genre,
        inputs="image",  # Entrée sous forme d'image
        outputs="text",  # Sortie sous forme de texte (le genre prédit)
        live=True,
        description="Upload an image to predict its genre using the trained model.",
    ).launch(server_port=7860, debug=True, share=True, server_name="0.0.0.0")
