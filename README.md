# Face Recognition
A PyTorch-based project that trains multiple CNN architectures to classify gender (Male/Female) from facial images.

## 🧠 Features
* Supports multiple CNN architectures: ResNet18, VGG16, and GoogLeNet  
* Includes transfer learning and fine-tuning capabilities  
* Provides model training and evaluation with accuracy and training time metrics  
* Offers image augmentation and visualization

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/danakembara/face_recognition.git
cd face_recognition

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Structure
Expected folder structure:
<pre>
  face_recognition/ 
  ├── dataset/ 
  │   ├── images/
  │   |   └── images.jpg
  │   └── labels/
  |       └── labels.txt
  ├── weights/
  |   └── weights_best.pth
  ├── models.py 
  ├── utils.py 
  ├── train.py 
  ├── evaluate.py 
  └── requirements.txt
</pre>

## 🏃 Usage
```bash
# Train the models
python train.py

# Evaluate the models
python evaluate.py
```

## 📈 Results
Performance at the best epoch:
| Model     | Train Accuracy (%) | Training Time (s) | Test Accuracy (%) |
|-----------|--------------------|-------------------|-------------------|
| ResNet18  | 97.88              | 399.08            | 95.76             | 
| VGG16     | 79.92              | 2522.19           | 78.53             |
| GoogleNet | 98.66              | 487.6             | 94.92             |

Future improvements:
* Increase dataset size (currently 1414 train and 354 test images) using additional data from the CelebA dataset
* Experiment with augmentation methods tailored specifically for gender classification
* Split dataset into train, validation, and test sets for model-specic hyperparameter tuning
* Explore alternative transfer learning techniques, such as freezing weights in different layers
* Investigate using smaller variants of VGG (e.g., VGG11 or VGG13) to improve accuracy and reduce training time

## 📦 Data Availability
This project uses a subset of the CelebA dataset, which is publicly available at the [official CelebA website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  

You can also download the specific subset used in this project here: [Google Drive link](https://drive.google.com/drive/folders/1Y-kRoMckL1pvxT2zFC_VC2yaD_v7Ph3k)
