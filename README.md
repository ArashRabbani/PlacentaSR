# PlacentaSR
## Resolution enhancement of placenta histological images using deep learning

In this repository, a method has been developed to improve the resolution of histological human placenta images. For this purpose, a paired series of high- and low-resolution images have been collected to train a deep neural network model that can predict image residuals required to improve the resolution of the input images. A modified version of the U-net neural network model has been tailored to find the relationship between the low resolution and residual images. After training for 900 epochs on an augmented dataset of 1000 images, the relative mean squared error of 0.003 is achieved for the prediction of 320 test images. The proposed method has not only improved the contrast of the low-resolution images at the edges of cells but added critical details and textures that mimic high-resolution images of placenta villous space. The availible dataset size is shrinked for practical reasons but the more complete version is sharable upon request. 

![](Images/Slide1.png)


## Citation
Arash Rabbani; Masoud Babaei, Resolution enhancement of placenta histological images using deep learning, Proceedings of the 4th International Conference on Statistics: Theory and Applications (ICSTA'22), Prague, Czech Republic – July 28- 30, 2022. 
