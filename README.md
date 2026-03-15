# data-science-test-task

## Overview

This repository contains the solution for a test assignments:

- **Machine Learning & OOP** (Task 1 – MNIST classification with interface)
- **Computer Vision + NLP pipeline** (Task 2 – NER + Image Classification + multimodal verification)

The project demonstrates:
- Clean OOP design with abstract interfaces
- Training and inference of transformer-based NER
- Fine-tuning of ResNet-18 for animal classification
- Multimodal consistency checking (text description vs image content)
- Handling of edge cases and basic negation logic

- ## Task 1 – MNIST Classification with OOP Interface

**Location**: `mnist-classifier/`

**Implemented models**:
- Random Forest (`MnistRandomForestClassifier`)
- Feed-Forward Neural Network (`NeuralNetClassifier`)
- Convolutional Neural Network (`CNNClassifier`)

All models implement the abstract interface `MnistClassifierInterface` with methods:
- `train(X, y)`
- `predict(X)`

Unified wrapper class: `MnistClassifier(algorithm)`  
Supported values: `"cnn"`, `"rf"`, `"nn"`

**Demo & examples**:  
→ `mnist_classifier_demo.ipynb`

## Task 2 – Multimodal Animal Verification (NER + Image Classification)

**Goal**: given text (e.g. "There is a cow in the picture") and an image → return `True` if the mentioned animal matches the one in the image (with basic negation handling).

**Components**:
- **NER model** — fine-tuned small transformer (`ner_animal_final/`) for extracting animal names
- **Image model** — ResNet-18 fine-tuned on Animals-10 dataset → 10 classes
- **Pipeline** — `pipeline.py` (main logic) + `inference_image.py` (image utils)

**Datasets**:
- NER: synthetic dataset (`labels.json`) with ~70–100 examples
- Images: Animals-10 (10 classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel)

**Main script**:  
`ner_image_classification/pipeline.py`

**Interactive demo & edge cases**:  
`text_image_match_demo.ipynb`

## Installation & Requirements

```bash
# 1. Clone repository
git clone <your-repo-url>
cd project

# 2. Create virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

How to Run
1. MNIST demo (Task 1)
Bashcd mnist-classifier
jupyter notebook mnist_classifier_demo.ipynb
2. Multimodal verification demo (Task 2)
Bash# Option A — interactive notebook (recommended)
jupyter notebook text_image_match_demo.ipynb

# Option B — command line script
cd ner_image_classification
python pipeline.py --text "There is a cat in the picture." --image "../images/cat.jpg"
