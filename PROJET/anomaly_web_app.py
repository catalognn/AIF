####### PARTIE 3 : INTERFACE WEB AVEC DETECTION D'ANOMALIES #######

import gradio as gr
from PIL import Image
import requests
import io
import os

#API_URL = os.getenv("API_URL", "http://genre_api:5000").strip('"')
API_URL = os.getenv("API_URL", "http://localhost:5000").strip('"')

# Fonction qui envoie l'image à l'API Flask et récupère la si c'est une anomalie ou non
def is_anomaly(image):
    
    image = Image.fromarray(image.astype('uint8'))
    
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    
    # Envoyer l'image en tant que données binaires à l'API Flask
    response = requests.post(API_URL + "/anomaly", files={"file": img_binary.getvalue()})
    
    # Si la requête est réussie, extraire la prédiction du genre
    if response.status_code == 200:
        prediction = response.json().get("is_a_poster", "Error: No prediction")
        return prediction  
    
    return f"Error: {response.status_code}"

# Interface Gradio
if __name__ == "__main__":
    gr.Interface(
        fn=is_anomaly,
        inputs="image",  # Entrée sous forme d'image
        outputs="text",  # Sortie sous forme de texte (le genre prédit)
        live=True,
        description="Upload an image to predict if it is a movie poster or not.",
    ).launch(server_port=7860, debug=True, share=True, server_name="0.0.0.0")
