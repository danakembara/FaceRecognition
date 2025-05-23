# Gender Classification
A PyTorch-based project that trains multiple CNN architectures to classify gender (Male/Female) from facial images.

## 🧠 Features
* Supports multiple CNN architectures: ResNet18, VGG16, and GoogLeNet  
* Includes transfer learning and fine-tuning capabilities  
* Provides model training and evaluation with accuracy and training time metrics  
* Offers visualizations for dataset samples and data augmentation  

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/danakembara/gender-classification.git
cd gender-classification

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Structure
Expected folder structure:
<pre>
  face_recognition/ 
  ├── dataset/ 
  │   ├── images/
  |   |   └── 000051.jpg
  |   |   └── ....
  │   └── labels/ 
  │       └── list_attribute.txt 
  ├── models.py 
  ├── utils.py 
  ├── train.py 
  ├── test.py 
  └── requirements.txt 
</pre>

## 🏃 Usage
```bash
# Train the models
python train.py

# Evaluate the models
python test.py
```

## 📈 Results

| Model          | Train Accuracy(%) | Training Time (s) | Test Accuracy(%) |
|----------------|-------------------|-------------------|------------------|
| ResNet18       | 97.88             | 399.08            | 95.76            | 
| VGG16          | 79.92             | 2522.19           | 78.53            |
| GoogleNet      | 98.66             | 487.6             | 94.92            |

