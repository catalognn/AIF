# api.py

from flask import Blueprint, request, jsonify
from torchvision import models, transforms
import torch
from PIL import Image
import io

# Define the API Blueprint
api_bp = Blueprint('api', __name__)

# Load your trained model weights
mobilenet = models.mobilenet_v3_small(pretrained=True)

N_genres = 10  # Number of classes
for param in mobilenet.features.parameters():
    param.requires_grad = False

model = torch.nn.Sequential(
    mobilenet.features,
    mobilenet.avgpool,
    torch.nn.Flatten(),
    torch.nn.Linear(576, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, N_genres)
)

# Load the trained weights (replace with your actual file path)
model.load_state_dict(torch.load('model_genre_classifier.pth'))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the REST API endpoint
@api_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image file
        image = Image.open(io.BytesIO(file.read()))
        image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Return the predicted class
        return jsonify({'class': int(predicted.item())})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
