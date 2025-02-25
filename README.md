# Deep Learning-Based Radar Target Classification

This project implements a deep learning model for classifying radar targets using micro-Doppler signatures. The model can distinguish between different targets like drones, birds, vehicles, and humans based on their radar signatures.

## Features
- Real-time radar target classification using deep learning
- Support for micro-Doppler spectrogram processing
- FastAPI backend for model deployment
- Streamlit UI for visualization
- Comprehensive data preprocessing pipeline

## Installation
```bash
git clone https://github.com/yourusername/Deep-Learning-Radar-Target-Classification.git
cd Deep-Learning-Radar-Target-Classification
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**
```bash
python src/data_preprocessing.py --data_dir data/raw --output_dir data/processed
```

2. **Model Training**
```bash
python train.py --config config/model_config.yaml
```

3. **Start API Server**
```bash
uvicorn api.main:app --reload
```

4. **Launch UI**
```bash
streamlit run api/ui.py
```

## Project Structure
- `data/`: Radar dataset storage
- `notebooks/`: Jupyter notebooks for exploration
- `src/`: Core model and processing code
- `models/`: Saved model checkpoints
- `api/`: FastAPI deployment code

## License
MIT

## Contributors
Elaheh Novinfard at elanvnfrd@gmail.com.
