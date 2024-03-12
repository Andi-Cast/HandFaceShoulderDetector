#pip install ultralytics
##to download library
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

#I downloaded 50 images from iStock.com and took 50 of my own. I then used CVAT to annotate the images.
#The annotation data was then downloaded in the YOLO 1.1 format. 

#File format for this 
#/data
#/data/images/train     ##contains raw images
#/data/labels/train     ##contains the YOLO 1.1 data

##'configuration.yaml' is a file created to direct the machine where the data used for training is located
##outputs training results
results = model.train(data = "configuration.yaml", epochs = 300)  # train the model
#This training took around 8 hours for 222 epochs. I really wanted to make sure the accuracy was high enough
#and the charts results were coming out well. 



