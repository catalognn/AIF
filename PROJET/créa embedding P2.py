import zipfile
import os
import pandas as pd
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import multiprocessing

# Fonction pour dézipper un fichier depuis un chemin local
def unzip_local_folder(zip_path, output_dir):
    if not os.path.exists(zip_path):
        print(f"Le fichier {zip_path} n'existe pas.")
        return
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Fichiers extraits dans : {output_dir}")


# Classe pour inclure les chemins dans les données
class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path

# Charger le modèle pré-entrainé pour extraire le vecteur
model = models.mobilenet_v3_small(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Créer le DataFrame avec les embeddings et les chemins
def create_embeddings_dataframe(dataset_path, batch_size=128):
    # Charger les données et créer le DataLoader
    dataset = ImageAndPathsDataset(dataset_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    
    paths_list = []
    
    # Extraire les embeddings pour chaque lot
    for x, paths in dataloader:
        with torch.no_grad():
            paths_list.extend(paths)
    
    # Créer un DataFrame
    df = pd.DataFrame({'path': paths_list})
    return df


# Définir le point d'entrée principal
if __name__ == '__main__':
    # Assurer la compatibilité avec Windows pour multiprocessing
    multiprocessing.set_start_method('spawn')  # Définit le mode de démarrage des processus

    zip_path = "movielens-20m-posters-for-machine-learning.zip"  # Remplacez par le chemin de votre fichier ZIP local
    output_dir = "movielens-20m-posters-for-machine-learning"  # Répertoire de sortie pour extraire les fichiers

    # Décompresser le fichier ZIP local
    unzip_local_folder(zip_path, output_dir)

    # Créer le DataFrame avec les embeddings
    df = create_embeddings_dataframe(output_dir)

    # Enregistrer le DataFrame au format CSV
    csv_path = "URLlist.csv"  # Spécifiez le chemin du fichier CSV
    df.to_csv(csv_path, index=True)  # Enregistrer sans l'index
    print(f"Le DataFrame a été enregistré dans : {csv_path}")
