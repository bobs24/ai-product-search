import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from qdrant_client import QdrantClient

# --- CONFIG ---
st.set_page_config(page_title="Voila Visual Search", layout="wide")
COLLECTION_NAME = "voila_products"

# --- CACHED RESOURCES ---
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=st.secrets["qdrant"]["url"],
        api_key=st.secrets["qdrant"]["api_key"]
    )

@st.cache_resource
def load_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

# --- LOGIC ---
client = get_qdrant_client()
model = load_model()

transform_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_embedding(image):
    img_t = transform_image(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_t).squeeze().tolist()
    return embedding

# --- UI ---
st.title("Voila.id Visual Search")
st.markdown("Upload a photo to find similar products in our database.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Your Upload", width=300)
    
    with col2:
        with st.spinner("Searching Catalog..."):
            # 1. Embed Query
            vector = get_embedding(image)
            
            # 2. Search Qdrant
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=12 
            )
            
            # 3. Display Grid
            if results:
                st.success(f"Found {len(results)} matches")
                grid_cols = st.columns(4)
                for i, hit in enumerate(results):
                    with grid_cols[i % 4]:
                        st.image(hit.payload["image_url"], use_column_width=True)
                        st.caption(f"SKU: {hit.payload['sku']}")
                        st.markdown(f"**{int(hit.score * 100)}% Match**")
            else:
                st.warning("No matches found.")