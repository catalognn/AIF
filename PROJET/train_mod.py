import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

# Définition de la classe personnalisée
class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path

# Préparation des données
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Charger le dataset
dataset = datasets.ImageFolder("C:\\Users\\lisec\\OneDrive\\Documents\\cours\\5A\\AIF\\PROJET_AIF\\content\\sorted_movie_posters_paligema", transform=transform)

# Diviser en jeux d'entraînement et de test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Charger MobileNetV3 Small pré-entraîné
mobilenet = models.mobilenet_v3_small(pretrained=True)
N_genres = len(dataset.classes)

# Geler les couches convolutives
for param in mobilenet.features.parameters():
    param.requires_grad = False

# Construire le modèle
model = nn.Sequential(
    mobilenet.features,
    mobilenet.avgpool,
    nn.Flatten(),
    nn.Linear(576, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, N_genres)
)

# Entraînement
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

# Sauvegarder les poids du modèle
model_path = "model_genre_classifier.pth"
torch.save(model.state_dict(), model_path)
print(f"Model weights saved to {model_path}")

# Fonction d'évaluation
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Évaluer le modèle sur le jeu de test
test_accuracy = evaluate_model(model, test_dataloader)
print(f"Test Accuracy: {test_accuracy:.2f}%")

##pour le moment : 46,14% d'accuracy
