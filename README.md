# Face Recognition
A PyTorch-based project that trains and evaluates multiple CNN architectures to classify gender (Male/Female) from facial images. The best-performing model, ResNet18, achieves high accuracy and efficiency, making it ideal for real-world use. For a detailed overview of the training and evaluation pipeline, please refer to the presentation file: face-recognition.pptx.

## 🧠 Features
* Supports multiple CNN architectures: ResNet18, VGG16, and GoogLeNet  
* Includes transfer learning and fine-tuning capabilities 
* Provides model training and evaluation with performance and efficiency metrics  
* Offers image augmentation and visualization

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/danakembara/face-recognition.git
cd face-recognition

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset
<pre>
  face_recognition/ 
  ├── dataset/ 
  │   ├── images/
  │   │   └── images.jpg
  │   └── labels/
  │       └── labels.txt
  ├── weights/
  │   └── weights_best.pth
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

## 📊 Model performance

<div align="center">

<table>
  <thead>
    <tr>
      <th style="text-align: center;">Model</th>
      <th style="text-align: center;">Accuracy</th>
      <th style="text-align: center;">Precision</th>
      <th style="text-align: center;">Recall</th>
      <th style="text-align: center;">F1-macro</th>
      <th style="text-align: center;">F1-weighted</th>
      <th style="text-align: center;">Training Time</th>
      <th style="text-align: center;">Inference time/image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;"><b>ResNet18</b></td>
      <td style="text-align: center;"><b>98%</b></td>
      <td style="text-align: center;"><b>0.968</b></td>
      <td style="text-align: center;"><b>0.984</b></td>
      <td style="text-align: center;"><b>0.979</b></td>
      <td style="text-align: center;"><b>0.98</b></td>
      <td style="text-align: center;"><b>26.61 min</b></td>
      <td style="text-align: center;"><b>0.043 s</b></td>
    </tr>
    <tr>
      <td style="text-align: center;">GoogleNet</td>
      <td style="text-align: center;">97.33%</td>
      <td style="text-align: center;">0.964</td>
      <td style="text-align: center;">0.971</td>
      <td style="text-align: center;">0.973</td>
      <td style="text-align: center;">0.973</td>
      <td style="text-align: center;">49.54 min</td>
      <td style="text-align: center;">0.048 s</td>
    </tr>
    <tr>
      <td style="text-align: center;">VGG16_fw</td>
      <td style="text-align: center;">90.53%</td>
      <td style="text-align: center;">0.888</td>
      <td style="text-align: center;">0.88</td>
      <td style="text-align: center;">0.902</td>
      <td style="text-align: center;">0.905</td>
      <td style="text-align: center;">82.33 min</td>
      <td style="text-align: center;">0.176 s</td>
    </tr>
  </tbody>
</table>

</div>

## 📦 Data Availability
This project uses a subset of the CelebA dataset, which can be downloaded here: [Google Drive link](https://drive.google.com/drive/folders/1BMcCUkpRA99ULUMWYUOboL19Yph2k3Vs). The CelebA dataset is publicly available at the [official CelebA website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
