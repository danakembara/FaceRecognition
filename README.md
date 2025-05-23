# Gender Classification
A PyTorch-based project that trains multiple CNN architectures to classify gender (Male/Female) from facial images.

## 🧠 Features
* Supports multiple CNN models (ResNet18, VGG16, GoogLeNet)
* Transfer learning and finetuning
* Model training and evaluation with accuracy and time metric
* Visualizations for dataset samples and augmentation

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/face-gender-classification.git
cd face-gender-classification

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Structure
Expected folder structure:
<pre> ```text 
  face_recognition/ 
  ├── dataset/ 
  │   ├── images/ 
  │   └── labels/ 
  │       └── list_attribute.txt 
  ├── models.py 
  ├── utils.py 
  ├── train.py 
  ├── test.py 
  └── requirements.txt 
  ``` </pre>

## 🏃 Usage
```bash
# Train the models
python train.py

# Evaluate the models
python test.py
```



