import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import warnings
import numpy as np
import importlib.metadata
from qdrant_client import QdrantClient, models
import io

# ------------------------------------------------------------------
# 1. BASIC SETUP
# ------------------------------------------------------------------

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Voila Visual Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLLECTION_NAME = "voila_products"

# ------------------------------------------------------------------
# 2. CUSTOM CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
    }
    div[data-testid="stImageCaption"] {
        font-size: 12px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. CACHED RESOURCES
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=st.secrets["qdrant"]["url"],
        api_key=st.secrets["qdrant"]["api_key"]
    )

@st.cache_resource(show_spinner=True)
def load_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    return model

# ------------------------------------------------------------------
# 4. INITIALIZATION
# ------------------------------------------------------------------

client = get_qdrant_client()
model = load_model()

transform_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def get_embedding(image: Image.Image) -> list:
    img_tensor = transform_image(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    # L2 Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()

def load_image(uploaded_file):
            try:
                img = Image.open(uploaded_file)
            except Exception:
                # fallback: read bytes directly (handles WebP in some environments)
                uploaded_file.seek(0)
                img = Image.open(io.BytesIO(uploaded_file.read()))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img

# ------------------------------------------------------------------
# 5. UI
# ------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Search Filters")
    limit = st.slider("Results count", 4, 20, 12, step=4)
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.60)

st.title("🛍️ Voila.id Visual Search")
st.markdown("##### Upload a product image to find similar items")

uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "webp"], label_visibility="collapsed")

if uploaded_file:
    col_l, col_r = st.columns([1, 3])
    
    with col_l:
        query_image = load_image(uploaded_file)
        st.image(query_image, use_container_width=True, caption="Query Image")
        
    with col_r:
        st.subheader("Similar Products")
        
        # 1. Embed
        vector = get_embedding(query_image)
        
        # 2. Search using query_points (The one function we know exists!)
        try:
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=limit,
                with_payload=True,
                score_threshold=threshold
            ).points
            
            # 3. Display
            if search_result:
                cols = st.columns(4)
                for i, hit in enumerate(search_result):
                    with cols[i % 4]:
                        score_pct = int(hit.score * 100)
                        color = "green" if score_pct > 80 else "orange" if score_pct > 60 else "red"
                        
                        st.image(hit.payload.get("image_url", ""), use_container_width=True)
                        st.markdown(f"**SKU:** `{hit.payload.get('sku', '-')}`")
                        st.markdown(f":{color}[**{score_pct}% Match**]")
            else:
                st.warning("No matches found.")
                
        except Exception as e:
            st.error(f"Search failed: {e}")