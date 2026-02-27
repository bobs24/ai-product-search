# AI Visual Search Engine

A sustainable, automated, and serverless AI-powered visual search engine for the Voila.id product catalog. This application decouples heavy processing from the user interface to ensure low latency and zero maintenance.

---

<details>
<summary><strong>📐 System Architecture</strong> (Click to Expand)</summary>

### 1. Automated Data Ingestion (Backend)

To maintain an up-to-date search index without manual intervention, the system utilizes a scheduled ETL pipeline.

- **Trigger**: GitHub Actions workflow runs daily at 02:00 WIB  
- **Extraction**: Connects to Metabase API (Question ID: 5971) to fetch current SKUs  
- **Processing**:
  - Checks for new SKUs against the Qdrant index
  - Downloads images and computes embeddings using **DinoV2 (ViT-S/14)**
- **Loading**: Upserts vectors and metadata into **Qdrant Cloud**

### 2. Real-Time Visual Search (Frontend)

A lightweight Streamlit web application.

- **Input**: User uploads an image  
- **Inference**: Generates a query vector using DinoV2  
- **Retrieval**: Performs Approximate Nearest Neighbor (ANN) search in Qdrant  
- **Output**: Displays visually similar products  

</details>

---

<details>
<summary><strong>🛠️ Technology Stack</strong></summary>

| Component | Technology | Reason for Choice |
|--------|------------|------------------|
| **AI Model** | DinoV2 (ViT-S/14) | Self-supervised, strong geometric understanding without labels |
| **Vector DB** | Qdrant Cloud | Serverless, high-performance ANN search |
| **Frontend** | Streamlit | Rapid Python-based UI development |
| **Automation** | GitHub Actions | Zero-maintenance scheduling |
| **Data Source** | Metabase | Direct BI integration |

</details>

---

<details>
<summary><strong>📂 Repository Structure</strong></summary>

```text
ai_product_search/
├── .github/
│   └── workflows/
│       └── daily_sync.yml        # 🤖 CI/CD: Runs every night at 2 AM
├── .streamlit/
│   └── secrets.toml              # 🔑 Credentials (ignored by Git)
├── src/
│   ├── __init__.py
│   ├── app.py                    # 📱 Frontend: Streamlit App
│   └── ingest_data.py            # ⚙️ Backend: Metabase → Qdrant ETL
├── requirements.txt              # 📦 Dependencies
├── .gitignore                    # 🛡️ Security rules
└── README.md                     # 📄 Documentation
