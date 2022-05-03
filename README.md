# What plane is that?

This is a personal interest project. The aim is develop and deploy a Keras image classification for the following plane models: 

+ Airbus: 320, 330, 340, 350, 380
+ Boeing: 737, 747, 777, 787

# Phase one: A380 v B747

The stage one of this project is to develop a binary model for A380 and B747 models only. This should be a much simpler task, and will give me an opportunity to set up the right computational infrastructure and anticipate any bottlenecks. 

Some major milestones to hit: 

1. Loading the data correctly and flexibly so that the results can be evaluated easily later. 
2. Use the MobileNet or other pre-trained model as the base model and modify it to suit this particular task. 
3. Produce some diagnostic statistics, including SHAP values on the images. 
4. Evaluate the classification results to ensure there isn't any inherent bias. 

Updates

+ 01 May 2022: A number of the downloaded images are not of the exterior of the plane itself, but vary between the interior or other parts of the plane. The `mobilenet_v2` model was used to make predictions on these images with the attempt to filter these out in later analysis.

# Phase 2: Extending model to handle multiple classes of plane models

This is the multi-class extension of stage 1. The reason that this is done is stages is the size of the data is much larger and will require longer training time. 

# Phase 3: Deployment

This will involve either some cloud technology for uploading of images, or, it can be put onto a mobile app. 
