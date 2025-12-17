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
import uuid
import os

def get_secret(key_group, key_name, env_name):
    try:
        # Try Streamlit Secrets first
        return st.secrets[key_group][key_name]
    except (FileNotFoundError, KeyError, AttributeError):
        # Fallback to Environment Variables
        return os.getenv(env_name)

# --- CONFIGURATION ---
QDRANT_URL = get_secret("qdrant", "url", "QDRANT_URL")
QDRANT_KEY = get_secret("qdrant", "api_key", "QDRANT_API_KEY")
MB_URL = get_secret("metabase", "url", "METABASE_URL")
MB_USER = get_secret("metabase", "username", "METABASE_USERNAME")
MB_PASS = get_secret("metabase", "password", "METABASE_PASSWORD")
MB_Q_ID = get_secret("metabase", "question_id", "METABASE_QUESTION_ID")
import requests

# 1. Setup
print("Initializing...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.to(device)
model.eval()

transform_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_metabase_data():
    print("Connecting to Metabase...")
    session = requests.post(f"{MB_URL}/api/session", json={"username": MB_USER, "password": MB_PASS})
    if session.status_code != 200: raise Exception(session.text)

    headers = {"X-Metabase-Session": session.json()['id']}
    response = requests.post(f"{MB_URL}/api/card/{MB_Q_ID}/query/json", headers=headers)

    df = pd.DataFrame(response.json())
    print(f"Found {len(df)} products.")
    return df

def process_fast(df, collection_name="voila_products"):
    collections = client.get_collections()
    exists = any(c.name == collection_name for c in collections.collections)

    if not exists:
        print(f"🆕 Creating NEW collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
    else:
        print(f"♻️  Found existing collection '{collection_name}'. Scanning for updates...")
    # -------------------------------------------------------------

    print("Processing images...")

    batch_size = 100
    total_skipped = 0

    # Process in Batches
    for i in tqdm(range(0, len(df), batch_size), desc="Batch Progress"):
        batch_df = df.iloc[i : i + batch_size]

        points_to_upload = []
        ids_to_check = []

        # 1. Generate UUIDs (Instant Math)
        for _, row in batch_df.iterrows():
            sku = str(row['SKU_UNIVERSAL'])
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, sku))
            ids_to_check.append(point_id)

        # 2. Check existence (Fast Lookup)
        existing_points = client.retrieve(
            collection_name=collection_name,
            ids=ids_to_check,
            with_payload=False,
            with_vectors=False
        )
        existing_ids = {p.id for p in existing_points}

        # 3. Filter Loop
        for idx, row in batch_df.iterrows():
            sku = str(row['SKU_UNIVERSAL'])
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, sku))

            # --- THE SPEED BOOST ---
            if point_id in existing_ids:
                total_skipped += 1
                continue # Skip download & embedding
            # -----------------------

            # (Rest of your download logic remains the same...)
            img_url = row['URL']
            if not img_url or pd.isna(img_url): continue

            try:
                response = requests.get(img_url, timeout=3)
                if response.status_code != 200: continue
                img = Image.open(BytesIO(response.content)).convert('RGB')

                img_t = transform_image(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(img_t).squeeze().cpu().numpy().tolist()

                points_to_upload.append(models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"sku": sku, "image_url": img_url}
                ))
            except:
                pass

        # 4. Upload
        if points_to_upload:
            client.upsert(collection_name=collection_name, points=points_to_upload)

    print(f"🎉 Done! Skipped {total_skipped} existing items.")

# RUN
if __name__ == "__main__":
    df_data = get_metabase_data()
    process_fast(df_data)
