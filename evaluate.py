import os
import torch
from models import googlenet, resnet18, vgg16
from utils import NoTransform, load_test_data, evaluate


if __name__ == "__main__":
    root = 'C:/Users/kemba/OneDrive/Desktop/Bootcamp/project/face_recognition/'
    weights_path = root + 'weights'

    batch_size = 32

    # Test
    test_loader = load_test_data(root, batch_size, transforms=NoTransform())

    model_resnet = resnet18()
    model_resnet.load_state_dict(torch.load(os.path.join(weights_path, 'resnet18_best.pth')))
    evaluate(model_resnet, test_loader)

    model_vgg = vgg16()
    model_vgg.load_state_dict(torch.load(os.path.join(weights_path, 'vgg16_best.pth')))
    evaluate(model_vgg, test_loader)

    model_googlenet = googlenet()
    model_googlenet.load_state_dict(torch.load(os.path.join(weights_path, 'googlenet_best.pth')))
    evaluate(model_googlenet, test_loader)