import matplotlib.pyplot as plt
from models import googlenet, resnet18, vgg16
from utils import (
    FaceRecognitionDataset, 
    CustomTransform, 
    NoTransform,
    visualize_dataset, 
    visualize_augmented_dataset, 
    count_parameters,
    load_train_data, 
    load_val_data,
    train
)


root = 'C:/Users/kemba/OneDrive/Desktop/Bootcamp/project/face_recognition/'

if __name__ == "__main__":

    # Hyperparameter
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.001
    
    # Loader
    train_loader = load_train_data(root, batch_size, transforms=CustomTransform(p=0.5))
    val_loader = load_val_data(root, batch_size, transforms=NoTransform())

    # Train
    train(model=resnet18(), 
          train_loader=train_loader, 
          val_loader=val_loader, 
          num_epochs=num_epochs, 
          learning_rate=learning_rate, 
          model_name='resnet18'
    )

    train(model=vgg16(), 
          train_loader=train_loader, 
          val_loader=val_loader,
          num_epochs=num_epochs, 
          learning_rate=learning_rate, 
          model_name='vgg16'
    )
    
    train(model=googlenet(), 
          train_loader=train_loader,
          val_loader=val_loader, 
          num_epochs=num_epochs, 
          learning_rate=learning_rate, 
          model_name='googlenet'
    )

# -------------------------------------------------------------------------------------
# --------------------- [OPTIONAL] EXPLORATORY ANALYSIS -------------------------------
# -------------------------------------------------------------------------------------

# # Investigate the number of images in dataset
# train_dataset = FaceRecognitionDataset(root, split='train')
# val_dataset = FaceRecognitionDataset(root, split='val')
# test_dataset = FaceRecognitionDataset(root, split='test')
    
# print(f"Number of training images: {len(train_dataset)}")
# print(f"Number of validation images: {len(val_dataset)}")
# print(f"Number of test images: {len(test_dataset)}")

# # Investigate the number of male and female in dataset
# all_labels = train_dataset.labels + val_dataset.labels + test_dataset.labels
# male_count = sum(all_labels)
# female_count = len(all_labels) - male_count
# plt.bar(['Female', 'Male'], [female_count, male_count], color=['skyblue', 'salmon'])
# plt.title('Gender Distribution in Dataset')
# plt.ylabel('Number of Images')
# plt.show()

# # Visualize examples of original and augmented images
# dataset_vis = FaceRecognitionDataset(root, split='train', transforms=NoTransform())
# visualize_dataset(dataset_vis, num_samples=5, randomize=True)
# visualize_augmented_dataset(dataset_vis, num_samples=5, randomize=True)

# # Count model's parameters
# model_resnet18 = resnet18()
# model_vgg16 = vgg16()
# model_googlenet = googlenet()
# count_parameters(model_resnet18, "ResNet18")
# count_parameters(model_vgg16, "VGG16")
# count_parameters(model_googlenet, "GoogleNet")