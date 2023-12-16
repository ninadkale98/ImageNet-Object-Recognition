import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torchvision
import torch
import datetime

import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from twilio.rest import Client
import psutil

account_sid = 'ACcd3189f80f4cd3cd244f34e7f00bd14c'
auth_token = '0f83bf0f3c8d827e69be25b92ef569d5'
client = Client(account_sid, auth_token)


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((227,227)),
    #torchvision.transforms.RandomResizedCrop(256, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

batch_size = 32
num_classes = 4


train_ds  = torchvision.datasets.ImageFolder( "Dataset/train", transform=transform_train) 
train_iter = torch.utils.data.DataLoader( train_ds, batch_size, shuffle=True, drop_last=True)

# Code Alexnet Model from scratch
class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), # 55 x 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 27 x 27
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # 27 x 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 13 x 13
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 13 x 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 13 x 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 13 x 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 6 x 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(), 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Load a pre-trained AlexNet model
model = AlexNet(num_classes=num_classes)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Training loop
max_num_epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Open text file to keep track of training progress
txt_cnt = 0
with open(f"training_progress.txt", "w") as f:
    f.write(f"Training Progress in {device}\n")


weight_dir = f"weights/{datetime.datetime.now().strftime('%H_%M_%S')}"

# create a directory to save weights with name date_HH_MM
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)


message = client.messages.create(
            body=f'Training Started',
            from_='+18666717304',
            to='+17169399055'
        )

print("Start training...")
last_saved = 0
last_loss = 1.5
epoch = 0
run = True
while run:
    try:
        last_saved +=1
        epoch += 1
        running_loss = 0.0
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # # Save Model parameters : Intermediate
            # if last_saved > 100 or last_loss - running_loss/len(train_iter) > 0.1:
            #     last_saved = 0
            #     last_loss = running_loss/len(train_iter)
                
            #     torch.save(model.state_dict(), f'{weight_dir}/model_epoch_{epoch}_file_{txt_cnt}_{datetime.datetime.now().strftime("%H%M%S")}.pth')
            #     txt_cnt += 1
                
            # Save Model : Terminate
            
            # if running_loss/len(train_iter) < 0.0001 :
            #     print("Loss is too low. Stopping training...")
            #     torch.save(model.state_dict(), f'{weight_dir}/weights_converged_with_loss_{round(running_loss/len(train_iter) , 7)}.pth')
            #     run = False
            #     with open(f"training_progress.txt", "a") as f:
            #         f.write(f"Epoch {epoch}/{max_num_epochs}, Loss: {running_loss/len(train_iter)}\n")
            #         f.write(f" Loss is too low, Stopping training")
            
        if epoch > max_num_epochs:
            print("Epoch limit reached. Stopping training...")
            torch.save(model.state_dict(), f'{weight_dir}/weights_max_epoch_loss_{round(running_loss/len(train_iter) , 7)}.pth')
            run = False
            with open(f"training_progress.txt", "a") as f:
                f.write(f"Epoch {epoch}/{max_num_epochs}, Loss: {running_loss/len(train_iter)}, CPU Load : {psutil.cpu_percent()}\n")
                f.write(f" Max epoch reached, Stopping training")
        
        if epoch % 10 == 0:
                print(f"Epoch {epoch}/{max_num_epochs}, Loss: {running_loss/len(train_iter)}")
                with open(f"training_progress.txt", "a") as f:
                    f.write(f"Epoch {epoch}/{max_num_epochs}, Loss: {running_loss/len(train_iter)}, CPU Load : {psutil.cpu_percent()} \n")
        
        if epoch % 200 == 0:
            message = client.messages.create(
            body=f'Learning : loss {round(running_loss/len(train_iter) , 7)} epoch {epoch}!',
            from_='+18666717304',
            to='+17169399055'
        )
            
    except KeyboardInterrupt:
        print(f"Epoch {epoch+1}/{max_num_epochs}, Loss: {running_loss/len(train_iter)}")
        print("Training stopped by user.")
        with open(f"training_progress.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{max_num_epochs}, Loss: {running_loss/len(train_iter)}, CPU Load : {psutil.cpu_percent()}\n")
            f.write("Training stopped by Keyboard Interruption.\n")
        torch.save(model.state_dict(), f'{weight_dir}/weights_keyboard_interrupt_epoch_{epoch}_loss_{round(running_loss/len(train_iter) , 7)}.pth') 
        run = False


# write code to change txt file name to training_progress_date.txt
os.rename("training_progress.txt", f"training_progress_{datetime.datetime.now().strftime('%d_%H_%M')}.txt")
print("Training complete.")

message = client.messages.create(
    body=f'Training complete: loss {round(running_loss/len(train_iter) , 7)} epoch {epoch} , CPU Load : {psutil.cpu_percent()}!',
    from_='+18666717304',
    to='+17169399055'
)

print(message.sid)