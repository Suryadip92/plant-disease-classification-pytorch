# Plant Disease Classification using PyTorch

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-brightgreen)](https://plant-disease-classification-be18.onrender.com/app)

## Overview

Plant Disease Classification using PyTorch is an end-to-end image-based solution for identifying plant leaf diseases and healthy leaves. It combines a PyTorch-trained model with a FastAPI backend and a responsive web frontend to provide real-time predictions from uploaded images or camera-captured photos.

## Live Demo

**[Open Live Application](https://plant-disease-classification-be18.onrender.com/app)**

## Key Features

- **Image Upload**: Upload plant leaf images (JPG, PNG, JPEG) via file picker or drag-and-drop
- **Camera Capture**: Capture photos directly using device camera with live preview
- **Real-time Prediction**: Get predicted class with confidence score and top-3 predictions
- **Disease Information**: View disease descriptions for predicted classes
- **Model Information**: Access detailed model stats and all 38 supported classes
- **Responsive UI**: Green-themed, animated, mobile-friendly interface

## Model Information

- **Architecture**: DynamicCNN (PyTorch)
- **Classes**: 38 plant disease and healthy-leaf classes
- **Input Size**: 256 × 256 × 3 RGB images
- **Preprocessing**: Resize to 256×256, convert to RGB, normalize with mean=0.5, std=0.5

## Technology Stack

- **ML/Deep Learning**: PyTorch, torchvision
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Image Processing**: Pillow, NumPy
- **Utilities**: joblib

## Project Structure

`
Plant-Disease-Classification/
├── artifacts/
│   ├── class_mapping.json
│   ├── model_config.json
│   ├── model.pt
│   ├── preprocessing_pipeline.joblib
│   └── training_history.json
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── notebooks/
│   └── plant_disease_classification.ipynb
├── plant_disease_inference.py
├── requirements.txt
└── .gitignore
`

## Live Application Usage

1. Visit the [live demo](https://plant-disease-classification-be18.onrender.com/app)
2. Choose **Upload Image** or **Capture from Camera**
3. Select or capture a plant leaf image
4. Click **Analyze Plant Disease**
5. View prediction results with confidence and details

## Run Locally

### Prerequisites
- Python 3.8+

### Setup

1. Clone the repository
`ash
git clone https://github.com/Suryadip92/plant-disease-classification-pytorch.git
cd plant-disease-classification-pytorch
`

2. Create and activate virtual environment (Windows)
`ash
python -m venv venv
venv\Scripts\activate
`

3. Install dependencies
`ash
pip install -r requirements.txt
`

4. Start the FastAPI server
`ash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
`

5. Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app)

## Model Artifacts

| File | Purpose |
|---|---|
| rtifacts/model.pt | Trained PyTorch model weights |
| rtifacts/preprocessing_pipeline.joblib | Image preprocessing pipeline |
| rtifacts/class_mapping.json | Class names and mappings |
| rtifacts/model_config.json | Model configuration and metadata |
| rtifacts/training_history.json | Training metrics history |

## Deployment

Deployed on **Render**: [https://plant-disease-classification-be18.onrender.com/app](https://plant-disease-classification-be18.onrender.com/app)

## Limitations

- Predictions depend on image quality and training dataset coverage
- Not a substitute for professional agricultural diagnosis

## Future Improvements

- Expand to more plant species
- Add treatment recommendations
- Improve model robustness

## Contributing

Contributions are welcome via Pull Requests.

## License

No license has been specified for this repository.
