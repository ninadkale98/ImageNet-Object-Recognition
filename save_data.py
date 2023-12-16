import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torchvision
import torch

import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


classes = ["chair" , "aeroplane" , "train" , "motorbike"]

annotation_directory_path = 'VOC2007_test/Annotations'
xml_files = os.listdir(annotation_directory_path)

for file in xml_files:
    xml_file_path = annotation_directory_path + '/' + file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Iterate through the XML elements and save cropped image   
    for object_elem in root.findall('.//object'):
        # Extract the class label
        class_label = object_elem.find('name').text
        if class_label.lower() in classes:
            # Extract the bounding box
            bndbox = object_elem.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Extract the image
            image_path = 'VOC2007_test/JPEGImages/' + file[:-4] + '.jpg'
            image = cv2.imread(image_path)
            
            # Crop the image
            cropped_image = image[ymin:ymax, xmin:xmax]
            
            # Save the image
            save_path = 'Dataset/test/' + class_label.lower() + '/' + file[:-4] + '.jpg'
            print(save_path)
            cv2.imwrite(save_path, cropped_image)