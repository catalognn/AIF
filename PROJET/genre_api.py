import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import torch.nn as nn
import gdown
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import os
from annoy import AnnoyIndex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GENRES = ['action', 'comedy', 'animation', 'documentary', 'drama', 'fantasy', 'horror', 'romance', 'science Fiction', 'thriller']

# Chemin vers les poids du modèle
MODEL_PATH = "model_genre_classifier1.pth"

# id du fihier à télécharger sur le drive
GOOGLE_DRIVE_FILE_ID = "1GfB_aUBAudIyBmUrbSAcYGwXd2KnBr64"

if not os.path.exists(MODEL_PATH):
    print("Model weights not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model = nn.Sequential(
    mobilenet.features,
    mobilenet.avgpool,
    nn.Flatten(),
    nn.Linear(576, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(GENRES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Initialisation de l'application Flask
app = Flask(__name__)

#Téléchargement depuis le drive de l'index Annoy
INDEX_PATH = "rec_imdb.ann"

# id du fihier à télécharger sur le drive
GOOGLE_DRIVE_FILE_ID2 = "1br1oGCqw9oTVqskHn8ZB2MOIjfHA-jfK"

if not os.path.exists(INDEX_PATH):
    print("Annoy index not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID2}", INDEX_PATH, quiet=False)

# Charger l'index Annoy
VECTOR_LENGTH = 576  #Longueur des vecteurs extraits du modèle
annoy_index = AnnoyIndex(VECTOR_LENGTH, 'angular')
annoy_index.load("rec_imdb.ann")  #téléchargement de l'index


#Téléchargement depuis le drive de le l'autoencoder
AUTOENCODER_PATH = "autoencoder_film_poster.pth"

# id du fihier à télécharger sur le drive
GOOGLE_DRIVE_FILE_ID3 = "1zaG-PGkSUUxabgHjL0Ei8rKeDMyNN5ne"

if not os.path.exists(AUTOENCODER_PATH):
    print("Autoencoder not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID3}", AUTOENCODER_PATH, quiet=False)

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # 224x224 -> 112x112
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 112x112 -> 56x56
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 56x56 -> 28x28
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 28x28 -> 14x14
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 112x112 -> 224x224
            nn.Sigmoid() # Normalize to [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=device))
autoencoder = autoencoder.to(device)
autoencoder.eval()



@app.route('/predict', methods=['POST']) #Route pour la PART1
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        img_pil = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Unable to process image: {str(e)}"}), 400

    tensor = transform(img_pil).to(device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_genre = GENRES[predicted.item()]

    return jsonify({"predicted_genre": predicted_genre})

@app.route('/reco', methods=['POST']) #Route pour la PART2
def recommend():
    vector = request.get_json()['vector']  # Récupérer le vecteur envoyé

    closest_indices = annoy_index.get_nns_by_vector(vector, 5) # Recherche des 5 films les plus proches

    # Retourner les chemins des films correspondants
    similar_movies = [closest_indices[i] for i in range(5)]

    return jsonify(similar_movies)

@app.route('/anomaly', methods=['POST'])#Route pour la PART3
def anomaly():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        img_pil = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Unable to process image: {str(e)}"}), 400

    tensor = transform(img_pil).to(device).unsqueeze(0)

    with torch.no_grad():
        reconstruction = autoencoder(tensor)
    
    error = torch.mean((tensor - reconstruction) ** 2, dim=[1, 2, 3])

    threshold = 0.001

    is_a_poster = error > threshold
    
    if is_a_poster :
        return jsonify({"is_a_poster": "Movie Poster"})
    else :
        return jsonify({"is_a_poster": "Not a Movie Poster"})

@app.route('/') #Route pour vérifier que l'API fonctionne
def home():   
    return 'Hello world!'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
