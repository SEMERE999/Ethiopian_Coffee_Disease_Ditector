# ☕ Coffee Doctor: Ethiopian Coffee Leaf Disease Detection System

An AI-powered diagnostic system and multilingual web application designed to identify Ethiopian coffee leaf diseases from images. 

This project implements a cascaded deep-learning architecture (Detector → Disease Classifier) and provides a fully integrated FastAPI backend and Progressive Web App (PWA) frontend. The system generates structured, professional advisory reports in English, Amharic (አማርኛ), Tigrinya (ትግርኛ), and Afaan Oromoo.

## System Architecture

The pipeline is split into two sequential machine learning models to ensure high accuracy and prevent false positives from non-leaf images:
1. **Coffee Leaf Detector:** A binary classifier that verifies if the uploaded image is an actual coffee leaf.
2. **Disease Classifier:** A multi-class model that categorizes the leaf as **Healthy, Phoma, Leaf Rust, or Cercospora**.

## Dataset & Training Artifacts

Due to GitHub's storage constraints, the large image dataset and the compiled `.keras` model artifacts are hosted externally on Google Drive. 

*  **[Raw Image Dataset]**(https://drive.google.com/drive/folders/1WnRrSZr5C9W83dNwELUjP3dMsjv6baQl)
*  **[Trained Models & Output Artifacts]**(https://drive.google.com/drive/folders/1grp9qceE-Z9jRM9cm-uojaL02RfO-JzA)

**System Architecture & Orchestration:**
* **Dataset:** The raw image dataset is structured into `detector/` (coffee vs. non-coffee) and `disease/` (the 4 disease classes) directories.
* **Training Orchestration:** Model training is executed via Google Colab (`Coffee_Doctor_Colab_Training.ipynb`) leveraging a T4 GPU for accelerated processing.
* **Output Artifacts:** During training, a custom background synchronization script automatically backs up all generated artifacts. This includes the compiled `.keras` model weights, evaluation reports, confusion matrices, and system logs.

## Technical Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Machine Learning** | TensorFlow / Keras | Core deep learning framework for training and inference. |
| **Backend API** | FastAPI, Uvicorn, Python | Asynchronous REST API serving the ML models and frontend. |
| **Frontend UI** | Vanilla HTML, CSS, JS | Lightweight, zero-build Progressive Web App (PWA). |
| **Environment** | Linux/Ubuntu, VS Code | Primary development and testing environment. |

## Project Structure
.
├── backend/
│   ├── app/                       # FastAPI web service layer
│   │   ├── main.py                # Application entrypoint
│   │   ├── knowledge/content.py   # Multilingual expert knowledge base
│   │   └── routers/               # API endpoints (predict, meta)
│   ├── src/EthiopianCoffee/       # Core ML training and processing pipeline
│   ├── final_model/               # Target directory for downloaded .keras models
│   ├── main.py                    # CLI for executing training pipelines
│   └── requirements.txt           # Python dependencies
├── frontend/                      # Vanilla JS Progressive Web App
│   ├── index.html                 # Main UI entrypoint
│   ├── sw.js, manifest.json       # Service worker for offline caching
│   └── static/                    # CSS, JS, Locales, and Icons
└── README.md


pass by a local agronomist or cooperative extension officer per language.
