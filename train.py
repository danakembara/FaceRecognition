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
    root = '..../face_recognition/'

    # Visualize dataset
    dataset_vis = FaceRecognitionDataset(root, is_train=False, transforms=NoTransform())
    
    # # Investigate the number of images in dataset
    # train_dataset = FaceRecognitionDataset(root, is_train=True)
    # test_dataset = FaceRecognitionDataset(root, is_train=False)
    # print(f"Number of training images: {len(train_dataset)}")
    # print(f"Number of testing images: {len(test_dataset)}")

    # # Visualize dataset
    # visualize_dataset(dataset_vis, num_samples=5, randomize=True) 

    # # Visualize augmented dataset
    # visualize_augmented_dataset(dataset_vis, num_samples=5, randomize=True)

    # Hyperparameter
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001

    # Train
    train_loader = load_train_data(root, batch_size, transforms=CustomTransform(p=0.5))

    train(resnet18(), train_loader, num_epochs, learning_rate, model_name='resnet18')
    train(googlenet(), train_loader, num_epochs, learning_rate, model_name='googlenet')
    
    num_epochs = 10
    train(vgg16(), train_loader, num_epochs, learning_rate, model_name='vgg16')
