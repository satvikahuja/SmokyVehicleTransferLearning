# Smoky Vehicle Detection Using Transfer Learning with ResNet-152

## NOTE: This is a basic prototype where only the smoky and non smoky vehicles are classified, need to add a ton of functionality 

## Overview
This project aims to detect vehicles emitting black or gray exhaust gases, which indicate a faulty exhaust system. It uses transfer learning applied to a ResNet-152 model pre-trained on the ImageNet dataset to differentiate between smoky and non-smoky vehicles.

## Motivation
Faulty vehicle exhausts are a significant contributor to air pollution and can cause various health problems. By detecting such vehicles, this project aids in environmental protection efforts and helps to enforce vehicle emission standards.

## Dataset
The model was trained on a custom dataset of 400 images labeled as 'smoky' or 'non-smoky'. This was created synthetically by https://github.com/srimantacse/VehicleSmokeDataset
To implement this project in future, need to make dataset from real world.

## Model
The model is based on ResNet-152, a deep residual network with 152 layers. We have done Transfer-Learing on the Resnet model with the smoky vehicle dataset to adapt the model to this specific task.

## Installation and Usage

Clone the repository to your local machine:

git clone https://github.com/your-username/smoky-vehicle-detection.git
cd smoky-vehicle-detection

### Prerequisites
Python , PyTorch, PIL, etc.

### Running the Model
python train_model.py

## Results
Have secured 91% accuracy on the test set
![pollution_2](https://github.com/satvikahuja/SmokyVehicleTransferLearning/assets/109898261/3dced20a-17e2-4ebd-a2bd-385b6f6dd5ca)


## Limitations and Future Work
Have to add multiple more features such as number plate detection, mapping the number plate with the vehicle detected. This is going to get quite complex to accomplish, will try to add these funcitonalities in the future

## Contributing
open to discuss in detail how we can make it production ready by how we could implement the future work

## Contact
satvik.ahuja13@gmail.com
