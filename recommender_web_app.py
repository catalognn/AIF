import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from annoy import AnnoyIndex
from transformers import DistilBertTokenizer, DistilBertModel
import gdown
import os

EMBEDDING_PATH = "metadata_embeddings.csv"

# id du fihier à télécharger sur le drive
EMBEDDING_FILE_ID = "1IqzZhEF86s4a15NdgYRLpBX2wOntIib2"

if not os.path.exists(EMBEDDING_PATH):
    print("Model weights not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}", EMBEDDING_PATH, quiet=False)

metadata = pd.read_csv(EMBEDDING_PATH)

#### BOW - Specific words similarity

# BOW tools
vectorizer = CountVectorizer(max_features=500)
vectorizer.fit(metadata["overview"])

# Load existing bow indexes
embedding_dim = 500 
bow_index = AnnoyIndex(embedding_dim, metric='angular')

BOW_ANNOY_PATH = "bow_annoy_index.ann"

# id du fihier à télécharger sur le drive
BOW_FILE_ID = "11roPBe-_afV7mTwvxdKyaCn9XhJIZ_dM"

if not os.path.exists(BOW_ANNOY_PATH):
    print("Model weights not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={BOW_FILE_ID}", BOW_ANNOY_PATH, quiet=False)
    
bow_index.load(BOW_ANNOY_PATH)

def get_bow_recommendations(query, top_n=5):
    query_embedding = vectorizer.transform([query]).toarray()[0]  # Convert query to BoW embedding
    neighbors = bow_index.get_nns_by_vector(query_embedding, top_n)  # Find nearest neighbors
    return metadata.iloc[neighbors][['original_title', 'overview']].to_dict(orient='records')

#### DISTILBERT - Semantic similarity 

# Distilbert tools
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load existing distilbert indexes
embedding_dim = 768
distilbert_index = AnnoyIndex(embedding_dim, metric='angular')

DISTILBERT_ANNOY_PATH = "distilbert_annoy_index.ann"

DISTILBERT_FILE_ID = "1y1PoxDC3LmdMzulYeUdbUdibdT5F2d5R"

if not os.path.exists(DISTILBERT_ANNOY_PATH):
    print("Model weights not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DISTILBERT_FILE_ID}", DISTILBERT_ANNOY_PATH, quiet=False)

distilbert_index.load(DISTILBERT_ANNOY_PATH)

def get_distilbert_recommendations(query, top_n=5):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]  # Convert query to DistilBERT embedding
    neighbors = distilbert_index.get_nns_by_vector(query_embedding, top_n)  # Find nearest neighbors
    return metadata.iloc[neighbors][['original_title', 'overview']].to_dict(orient='records')

# Function to format the recommendations for display
def format_recommendations(recommendations):
    formatted = ""
    for idx, rec in enumerate(recommendations):
        title = f"**{rec['original_title']}**"
        overview = rec['overview']
        formatted += f"# {idx+1}. {title} \n### {overview}\n---\n"
    return formatted

# Gradio interface function
def recommend_movies(query, method):
    if method == "Bag-of-Words":
        recommendations = get_bow_recommendations(query)
    elif method == "DistilBERT":
        recommendations = get_distilbert_recommendations(query)
    return format_recommendations(recommendations)

# Create Gradio interface
interface = gr.Interface(
    fn=recommend_movies,
    inputs=[
        gr.Textbox(label="Enter a movie description/plot", placeholder="Type your movie description here..."),
        gr.Radio(["Bag-of-Words", "DistilBERT"], label="Choose Recommendation Method")
    ],
    outputs=gr.Markdown(label="Movie Recommendations"),
    title="Movie Recommendation System",
    description="Enter a movie description, select a method (Bag-of-Words or DistilBERT), and get 5 recommended movies."
)

# Launch the app
interface.launch()
