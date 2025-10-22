# 🦵 Knee X-Ray Classification using Deep Learning

This repository contains a deep learning pipeline for classifying knee X-ray images into multiple categories using **DenseNet201** and **MobileNetV2** backbones.  
The codebase is modular, separating **preprocessing**, **modeling**, and **training** stages for better maintainability and scalability.

---

## 📁 Project Structure

knee-xray-classification/
│
├── data/
│ └── dataset/ # X-ray images (excluded from Git)
│
├── notebooks/
│ └── exploratory.ipynb # Optional for data exploration
│
├── src/
│ ├── preprocessing.py # Image loading, cleaning, augmentation
│ ├── modeling.py # Model creation and training
│ └── init.py
│
├── main.py # Entry point to run the full pipeline
├── requirements.txt # Python dependencies
└── README.md # Documentation


---

## 🧠 Features

- Advanced preprocessing:
  - Pixel correction and edge sharpening  
  - Morphological cleaning and knee joint segmentation  
  - Augmentation: rotation, flipping, and zooming  
- Dual CNN architecture (DenseNet201 + MobileNetV2)  
- Rich evaluation metrics: Accuracy, AUC, Precision, Recall, F1-Score  
- Learning rate scheduling via `ReduceLROnPlateau`  
- Optional Grad-CAM visualization for interpretability  

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/knee-xray-classification.git
cd knee-xray-classification

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt 

## 🚀 Usage

Run the end-to-end pipeline:
python main.py

## 📚 Citation

If you use this repository or any part of the codebase in your research or project, please cite it as follows:

### 🔖 BibTeX

```bibtex
@article{shahid12potential,
  title={Potential of AI-Based Diagnostic Grading System for Knee Osteoarthritis (KOA)},
  author={Shahid, Saman and Wali, Aamir and Javaid, Aatir and Zikria, Shahid and Osman, Onur and Rasheed, Jawad},
  journal={Frontiers in Medicine},
  volume={12},
  pages={1707588},
  publisher={Frontiers}
}

