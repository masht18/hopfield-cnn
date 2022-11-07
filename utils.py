# -*- coding: utf-8 -*-
import torch
import numpy as np

def generate_label_reference(dataset, num_classes=10):
    
    '''
    Creates a label reference of size [num_classes x num_images_per_class] for sequence generation.
    This label reference helps find an example of a specific class given a dataset.
        :param dataset (torchvision.datasets):
            dataset to generate label reference from
        :return label_ref(tensor):
            indices of images with given class for each class 
        
    '''
    labels = torch.tensor(dataset.targets)
    label_ref = []

    for i in range(num_classes):
        label_ref.append((labels == i).nonzero().squeeze())
        
    return label_ref
