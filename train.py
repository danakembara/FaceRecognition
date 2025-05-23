import os
import matplotlib.pyplot as plt
import torch
from models import googlenet, resnet18, vgg16
from utils import (
    FaceRecognitionDataset, 
    CustomTransform, 
    NoTransform, 
    visualize_dataset, 
    visualize_augmented_dataset, 
    load_train_data, 
    train
)


if __name__ == "__main__":
    root = 'C:/Users/kemba/OneDrive/Desktop/Bootcamp/project/face_recognition/'

    # Visualize dataset
    dataset_vis = FaceRecognitionDataset(root, is_train=False, transforms=NoTransform())

    # visualize_dataset(dataset_vis, num_samples=5, randomize=True) 

    # # Visualize augmented dataset
    # visualize_augmented_dataset(dataset_vis, num_samples=5, randomize=True)

    # Hyperparameter
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Train
    train_loader = load_train_data(root, batch_size, transforms=CustomTransform(p=0.5))

    train(resnet18(), train_loader, num_epochs, learning_rate, model_name='resnet18')
    train(vgg16(), train_loader, num_epochs, learning_rate, model_name='vgg16')
    train(googlenet(), train_loader, num_epochs, learning_rate, model_name='googlenet')
