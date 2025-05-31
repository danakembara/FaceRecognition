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
git clone https://github.com/danakembara/face-recognition.git
cd face-recognition

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Structure
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

## 📈 Results and Discussion
<div align="center">
Table 1. Models performance evaluated on the test set:
</div>

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


## 📈 Conclusion

## 🚀 Future Improvements
* Increase dataset size (currently 1414 train and 354 test images) using additional data from the CelebA dataset
* Experiment with augmentation methods tailored specifically for gender classification
* Split dataset into train, validation, and test sets for model-specic hyperparameter tuning
* Explore alternative transfer learning techniques, such as freezing weights in different layers
* Investigate the use of smaller VGG variants (e.g., VGG11 or VGG13) to improve accuracy and reduce training time, especially since the current model only trains for 10 epochs while others are trained for 30

## 📦 Data Availability
This project uses a subset of the CelebA dataset, which is publicly available at the [official CelebA website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  

You can also download the specific subset used in this project here: [Google Drive link](https://drive.google.com/drive/folders/1Y-kRoMckL1pvxT2zFC_VC2yaD_v7Ph3k)
