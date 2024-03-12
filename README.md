# **My First Detector**
### This was my final project for my Biometrics class.
-----
###### **Given this is part of an assignments from a course, I do not condone plagarism. Please don't copy my code and claim as yours.**
-----
## **Assignment Details**
- Purpose: Design a program that is able to detect hands, shoulder, and hands.Another feature that should be included was dectect when the shoulders were shrugging. 
- Given: Since this was an open project, the only thing discussed prior to this assignment were methods on how to achieve the goal of this assignment.
- Notes: I decided to use Python as my programming language of choice and used the YOLO to train a classifier for the hands. This was my first time training my own custom detector.
-----

## **Examples Images**
###### Below I have included images from test I ran using my classifer. 
- Image 1: This is an example of a clear detection of all features.
<img src="https://github.com/Andi-Cast/Hand_Face_Shoulder_Detector/blob/main/ExampleScreenShots/clearDetection.png" height="auto" width="50%" >
- Image 2: This image displays valid detections but due to the hands obstucting the shoulder regions there were error with detections.
<img src="https://github.com/Andi-Cast/Hand_Face_Shoulder_Detector/blob/main/ExampleScreenShots/shoulderDetectionErrorOne.png" height="auto" width="50%" >
- Image 3: This is a classic problem that was expected with this assignment. It is very challenging developing a program that is able to detect every feature when the hands were placed on/behind the head. To my surprise, the detector was able to find the hands in this frame.<img src="https://github.com/Andi-Cast/Hand_Face_Shoulder_Detector/blob/main/ExampleScreenShots/handsOneHeadError.png" height="auto" width="50%" >

-----
## **Steps I Took To Train Custom Classifier**
- 1. Find A Training Model: There are various models that can used to train your own classifier. I decided on YOLO which was popular when using Python. 
- 2. Collecting Data & Formatting: In order to collect data to feed the classifier I used downloaded images from the internet and took screenshots from the video I would be using to test the classifer. Each image whould then be annotated to box in where the hands were in each image. The data used for this project can be found under the data/ directory. 
- 3. Running the Training: Once all the data was collected, the training was ready to begin. I have included the file I used to do this (HandTraining.py). This requires a lot of processing power and it took a couple of hours for me because I was running it on a Macbook Air.
- 4. Testing: The final part was testing. I would run the classifier on a video I took and if the results weren't the best it meant I had to either collect more data points along with trainging the classifier a bit longer.

-----
