# What plane is that?

This is a personal interest project. The aim is develop and deploy a Keras image classification for the following plane models: 

+ Airbus: 320, 330, 340, 350, 380
+ Boeing: 737, 747, 777, 787

# Stage one: A380 v B747

The stage one of this project is to develop a binary model for A380 and B747 models only. This should be a much simpler task, and will give me an opportunity to set up the right computational infrastructure and anticipate any bottlenecks. 

Some major milestones: 

1. Loading the data correctly and flexibly so that the results can be evaluated easily later. 
2. Use the MobileNet model as the base model and modify it to suit this particular task. 
3. Produce enough diagnostic statistics, including SHAP values on the images. 
4. Evaluate the classification results to ensure there isn't any inherent bias. 

Stage 2: Deployment of Stage 1 model

This will involve either some cloud technology for uploading of images, or, it can be put onto a mobile app. 

Stage 3: Extending model to handle multiple classes of plane models

This is the multi-class extension of stage 1. The reason that this is done is stages is the size of the data is much larger and will require longer training time. 

Stage 4: Deployment of Stage 3 model