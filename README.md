# 🏥 Kidney Disease Classification (End-to-End MLOps Pipeline)

An enterprise-grade, end-to-end Deep Learning application designed to classify Kidney CT scan images into four clinical categories: **Cyst, Normal, Tumor, and Stone**. 

Rather than being just a standalone model script, this project is built on **MLOps best practices**—featuring rigorous data versioning, modular pipelines, pipeline reproducibility, real-time experiment tracking, and clean frontend/backend serving architectures.

---

## 🏗️ System MLOps Architecture

This pipeline is engineered to solve the classic "it works on my machine" problem by linking the absolute best MLOps tools together:

1. **DVC (Data Version Control):** Tracks raw CT scan datasets and model weights in external storage (e.g., DagsHub, S3, or GDrive) without bloating your Git history.
2. **MLflow Tracking:** Logs hyperparameters (epochs, learning rate) and tracking metrics (`accuracy: 81.80%`, `loss: 3.8583`) dynamically during your runs.
3. **Docker:** Packages your training code, the finalized `.h5` model, the FastAPI backend, and the Streamlit frontend into an isolated environment to ensure absolute runtime consistency.

---

## 📊 Performance Metrics

| Metric | Value | Status |
| :--- | :---: | :---: |
| **Loss** | `3.8583` | Stable |
| **Validation Accuracy** | **`81.80%`** | **Production-Ready** |

---

## 🗺️ Pipeline Workflow

```text
[ Raw Data Source ]
       │
       ▼ (DVC Stage 1)
┌─────────────────────────┐
│     Data Ingestion      │ ──> Download & Unzip Raw CT Scans
└─────────────────────────┘
       │
       ▼ (DVC Stage 2)
┌─────────────────────────┐
│   Prepare Base Model    │ ──> Load VGG16 & Modify Output Classes (4 Classes)
└─────────────────────────┘
       │
       ▼ (DVC Stage 3)
┌─────────────────────────┐
│     Model Training      │ ──> Train with Augmentation & Log Metrics to MLflow
└─────────────────────────┘
       │
       ▼ (DVC Stage 4)
┌─────────────────────────┐
│    Model Evaluation     │ ──> Calculate Loss/Accuracy & Produce Metrics JSON
└─────────────────────────┘
       │
  ┌────┴──────────────────────────┐
  ▼ (FastAPI Service)             ▼ (Streamlit Web Interface)
┌─────────────────────────┐     ┌─────────────────────────┐
│     Backend Engine      │ <──> │      Frontend App       │
│ (Serves Predictions)    │      │  (User Image Upload)    │
└─────────────────────────┘     └─────────────────────────┘
```

---

## 🧪 MLflow Experiment Tracking & Configuration

Your training runs and evaluation metrics are logged seamlessly using **MLflow**. Follow these configurations to track your runs locally or on a remote platform like DagsHub.

### 1. Set up MLflow Credentials (If using a remote tracker like DagsHub)
Set your environment variables in your terminal before training:

```powershell
# In Windows PowerShell:
$env:MLFLOW_TRACKING_URI="[https://dagshub.com/your-username/your-repo.mlflow](https://dagshub.com/your-username/your-repo.mlflow)"
$env:MLFLOW_TRACKING_USERNAME="your-username"
$env:MLFLOW_TRACKING_PASSWORD="your-dags-hub-token-or-password"
```

```bash
# In Bash/Linux/macOS:
export MLFLOW_TRACKING_URI="[https://dagshub.com/your-username/your-repo.mlflow](https://dagshub.com/your-username/your-repo.mlflow)"
export MLFLOW_TRACKING_USERNAME="your-username"
export MLFLOW_TRACKING_PASSWORD="your-dags-hub-token-or-password"
```

### 2. Launch the Local MLflow UI Dashboard
If you are tracking experiments on your local computer, open a separate terminal and execute:

```bash
mlflow ui
```
Once run, open your browser and navigate to **`http://127.0.0.1:5000`** to view performance graphs, hyperparameter parameters, and training curves in real-time.

---

## 🛠️ Tech Stack

* **Core AI**: TensorFlow 2.x, Keras (VGG16)
* **MLOps**: DVC, MLflow
* **CI/CD**: GitHub Actions (`.github/workflows/ci-cd.yaml`)
* **Containerization**: Docker
* **Deployment Services**: FastAPI, Streamlit, Uvicorn

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository & Initialize Environment
```bash
git clone [https://github.com/your-username/Kidney-Disease-Classification-DLOPS.git](https://github.com/your-username/Kidney-Disease-Classification-DLOPS.git)
cd Kidney-Disease-Classification-DLOPS

python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate
```

### 2. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the DVC Pipeline
Execute the pipeline stages with one simple execution command:
```bash
dvc repro
```

### 4. Spin up the App UI Services
Open two terminal instances side-by-side:

* **Terminal 1 (Backend API Service):**
    ```bash
    python app.py
    ```
* **Terminal 2 (Frontend Streamlit Dashboard):**
    ```bash
    streamlit run frontend.py
    ```
Navigate to `http://localhost:8501` to use the classification portal!

---

## 🐳 Docker Deployment

To build and run your services using Docker containerization:

### 1. Build the Docker Image
```bash
docker build -t kidney-disease-classifier .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 -p 8501:8501 kidney-disease-classifier
```
###📞 Contact & Portfolio
Developed with care by a software developer focused on robust machine learning systems.

Email: gargeesharma52@gmail.com
* **LinkedIn**: https://www.linkedin.com/in/gargee-sharma6548
---
