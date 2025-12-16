import streamlit as st
import requests
import pandas as pd
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.http import models
from torchvision import transforms
from tqdm import tqdm
import os

# 1. Load Secrets
def get_secret(key_group, key_name, env_name):
    try:
        # Try Streamlit Secrets first
        return st.secrets[key_group][key_name]
    except (FileNotFoundError, KeyError, AttributeError):
        # Fallback to Environment Variables
        return os.getenv(env_name)

# Load the configuration
QDRANT_URL = get_secret("qdrant", "url", "QDRANT_URL")
QDRANT_KEY = get_secret("qdrant", "api_key", "QDRANT_API_KEY")
MB_URL = get_secret("metabase", "url", "METABASE_URL")
MB_USER = get_secret("metabase", "username", "METABASE_USERNAME")
MB_PASS = get_secret("metabase", "password", "METABASE_PASSWORD")
MB_Q_ID = get_secret("metabase", "question_id", "METABASE_QUESTION_ID")

# Check if we successfully loaded credentials
if not QDRANT_URL or not MB_PASS:
    raise ValueError("CRITICAL ERROR: Credentials not found in secrets.toml OR Environment Variables.")

# 2. Setup Qdrant & Model
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.to(device)
model.eval()

# Image Preprocessing
transform_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_metabase_data():
    print("Connecting to Metabase...")
    session_url = f"{MB_URL}/api/session"
    payload = {"username": MB_USER, "password": MB_PASS}
    
    # Login
    session = requests.post(session_url, json=payload)
    if session.status_code != 200:
        raise Exception(f"Metabase Login Failed: {session.text}")
    
    token = session.json()['id']
    headers = {"X-Metabase-Session": token}
    
    # Get Question Data
    print(f"Fetching Question {MB_Q_ID}...")
    query_url = f"{MB_URL}/api/card/{MB_Q_ID}/query/json"
    response = requests.post(query_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")
        
    df = pd.DataFrame(response.json())
    print(f"✅ Found {len(df)} products.")
    return df

def setup_collection():
    collection_name = "voila_products"
    
    # Check if exists, if not create
    collections = client.get_collections()
    exists = any(c.name == collection_name for c in collections.collections)
    
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Using existing collection: {collection_name}")
    
    return collection_name

def process_and_upload(df, collection_name):
    print("Processing images... this might take a while.")
    
    batch_size = 50
    points = []
    
    # Loop through DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        sku = str(row['SKU_UNIVERSAL'])
        img_url = row['URL']
        
        # Sustainable: Skip if URL is empty
        if not img_url or pd.isna(img_url):
            continue
            
        try:
            # 1. Download Image
            response = requests.get(img_url, timeout=5)
            if response.status_code != 200: continue
            
            img = Image.open(BytesIO(response.content)).convert('RGB')

            # 2. Embed
            img_t = transform_image(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(img_t).squeeze().cpu().numpy().tolist()
            
            # 3. Prepare Point
            points.append(models.PointStruct(
                id=index, # Or hash of SKU
                vector=embedding,
                payload={"sku": sku, "image_url": img_url}
            ))
            
        except Exception as e:
            # print(f"Skipping {sku}: {e}")
            pass
        
        # Upload Batch
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []

    # Upload remaining
    if points:
        client.upsert(collection_name=collection_name, points=points)
    print("Upload Complete!")

if __name__ == "__main__":
    df = get_metabase_data()
    col_name = setup_collection()
    process_and_upload(df, col_name)