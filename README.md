# Stable Diffusion Image Generator for Medical Aesthetics

A GPU-accelerated image generation system using FastAPI, PyTorch, and Stable Diffusion for visualizing medical aesthetic treatments.

## Features

- GPU-accelerated image generation with CUDA support
- ControlNet integration for better feature preservation
- Support for both Botox and Filler treatments
- Automatic memory management and optimization
- RESTful API interface
- Comprehensive error handling and validation

## Requirements

- NVIDIA GPU with CUDA support
- CUDA 12.1 or later
- Python 3.10 or later

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
cd sdig/app
python main.py
```

2. The server will start on `http://localhost:8000`

## API Endpoints

### GET /health
Health check endpoint

### GET /areas
Get available treatment areas for both Botox and Fillers

### POST /generate
Generate treatment visualization

Parameters:
- `file`: Image file (multipart/form-data)
- `area`: Treatment area (form field)
- `injection_number`: Number of units for Botox treatments (form field, optional)

## Treatment Areas

### Botox Areas
- Forehead Lines
- Frown Lines (Glabella)
- Crow's Feet
- Nasalis Lines
- Vertical Lip Lines
- Lip Flip
- Smile Lift
- Masseter Reduction
- Dimpled Chin
- Platysmal Bands

### Filler Areas
- Cheek
- Smile Line
- Lip
- Temple
- Nose

## Technical Details

The system uses:
- Stable Diffusion 2.1 as the base model
- ControlNet for feature preservation
- TF32 optimizations on Ampere GPUs
- xFormers for memory efficiency
- Attention slicing for reduced memory usage
- Dynamic strength calculation based on injection units
- Automatic image preprocessing and validation

## Error Handling

The system includes comprehensive error handling for:
- Invalid input images
- Unsupported treatment areas
- GPU memory issues
- Model loading failures
- Invalid parameters

## Memory Management

The system implements several memory optimization techniques:
- Automatic CUDA cache clearing
- Garbage collection after generation
- Memory-efficient attention mechanisms
- Dynamic batch size adjustment
- Pipeline optimization with ControlNet 