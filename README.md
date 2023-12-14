# INTRODUCTION
We strive to display the one of the various applications of Machine learning. We have incorporated various python libraries to achieve that. The data has 588 images and 588 masks for each image. The masks highlight the parts of the leaf which have been infected or have died. This dataset includes diverse diseases like Apple Scab, Apple Rust, Corn Leaf Blight. The aim of this project is to create a model which can perform image segmentation and create a mask for any given image.

## Masked                             vs                           Original

![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/c7d297b2-ef35-4c77-ad4b-b520bc0e1250)
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/ab7f88c0-cf77-4e11-afc9-e6455c6c5bdd)

# PROBLEM STATEMENT & OBJECTIVES
The Problem statement – Create a machine learning model which can perform image segmentation for a leaf.
The Prime Objective is to isolate the diseased part with the highest possible accuracy.

# METHODOLOGY
The first step to achieving this goal was to preprocess our images and make them accessible to CNN’s and algorithms. For this we have used python libraries like Numpy, Matplotlib. Numpy was used to augment the images into numpy arrays which we then fed into the neural networks. Matplotlib was utilized to plot images and masks. We have also used OpenCV to further augment our images for the K-means Clustering algorithm.

We chose to use CNNs because they are apt for processing images and do not require as much processing power as ANN’s. We have implemented two neural networks : 

## 1. U-Net 

Olaf Ronneberger et al. Have designed this neural network and it is well known for its ability to perform image segmentation.
The architecture of U-Net is unique in that it consists of a contracting path and an expansive path. The contracting path contains encoder layers while the expansive path contains decoder layers that decode the encoded data and use the information from the contracting path via skip connections to generate a segmentation map.
This architecture makes the image smaller and then larger again using the previous information provided by the skip connections. This helps reduce the computational cost and hence makes the model train faster.

![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/9cbb2677-bf72-4048-8e09-eb42facccf55)

## 2.	Our custom neural network

This neural network is made by using the TensorFlow library. It creates a neural network layer by layer; we have taken inspiration from the U-net but we have made it quite smaller and have removed the skip connections. This model performs well but does not have the versatility that the U-net provides

## 3. K- means Clustering
We have also used K-means clustering to show how much more powerful and flexible CNNs are compared to a machine learning algorithm. K-means divides the image in K clusters with each cluster having the average color of all the pixels in that cluster. This essentially segments the image and sets the diseased segment apart.

# RESULTS & SNAPSHOTS

In this project we have implemented two neural networks and one machine learning algorithm. This has helped us highlight the various applications of AI and how neural networks can be used to enhance the versatility of a model.

# Results for K-means clustering :
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/d4a0ea09-b78d-4b48-9ce9-8b1c2d661330)

# Results for U-net :
## Predicted Mask               vs           Original Image
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/8eba839d-7226-4506-bc9f-99fa771e900c)
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/c8f53843-f853-49a3-9635-f586a60f3dad)
## Accuracy       vs        Epoch
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/494773c6-0781-4101-a951-aee86a7347a0)
# Results for our custom Neural Network :
## Original Image                  vs                        Masked Image
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/4651d24c-080d-46f5-b01c-d340aa8a559c)
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/525ad809-af99-42b1-ba68-f6916ce956a3)
## Accuracy       vs        Epoch
![image](https://github.com/Apetun/Leaf-disease-segmentation-using-Unet/assets/114131523/e4a0c564-b3af-4f5d-b40e-816c9409cd93)
# CONCLUSION
Through this project we have learned how versatile CNNs are and how we can also use machine learning algorithms to perform tasks like image segmentation even if they are not perfect, they work well. We have made use of a lot of the things taught in this course which has helped us complete this project.
# LIMITATIONS & FUTURE WORK
We believe that there is no perfect model and every model can be made better for its use case. Hance we think that the model we used can be made better and even the dataset can be made better, we can incorporate a larger variety of images and we can add or remove some layers from our neural network, we can also experiment with the loss functions and see which one is the best. 

We believe that this project has a use case in the real world, so if we can increase the accuracy of this model then we can use it to detect leaf diseases of a plant which can in turn help farmers rectify the problem and increase the lifespan of that plant and the yield.

# References
1. https://arxiv.org/abs/1505.04597
2. https://www.tensorflow.org/guide
3. https://docs.opencv.org/4.x/
4. https://pandas.pydata.org/docs/
5. https://numpy.org/doc/
6. https://matplotlib.org/stable/index.html











