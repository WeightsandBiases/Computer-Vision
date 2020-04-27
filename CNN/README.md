# Classification of House Numbers in Images using Convolutional Neural Networks

### Project Description

This project consists of a digit detection system which takes a sequence of images and returns any sequence of digits visible in the images.  It aims to solve the challenges in computer vision and image classification by overcoming the following common obstacles:

- Scale Invariance - Categorize numbers of different scales in the image
- Location Invariance - Categorize numbers of different locations in the image
- Font Invariance - Categorize numbers of different fonts
- Pose Invariance - Categorize numbers at different camera angles
- Lighting Invariance - Categorize numbers at different lighting conditions
- Noise Invariance - Categorize numbers in images that have Gaussian noise

### High Level Architecture

![alt text](./readme_images/Architecture.png)

### Dependencies

- tensorflow-gpu = 2.1.0
- Python > 3.6.7
- scikit-learn = 0.20.0
- h5py = 2.9.0
- opencv-python = 3.4.3.18

### File Structure

This repository contains the following

- README.md - Instruction file you are currently reading :)
- cnn_house_numbers.py - Tensorflow based CNN model code used to train and save the CNN model using the 32x32 SVHN dataset found here http://ufldl.stanford.edu/housenumbers/.
- house_number_detector.py - The pipeline which preprocesses common images and uses the Tensorflow based CNN model to detect numbers in the image.
- run.py - The executable to run all the stages of the image recognition
- training_images (directory) - consists of 32x32 images in the SVHN dataset that is saved in matlab format
- input_images(directory) - consists of sample images to be run in the pipeline, any additional images for number recognition shall be placed here
- readme_images(directory) - consists of images that are in this README
- tf_logs (runtime directory) - consists of tf logs
- tf_model (runtime directory) - consists of model and processed SVNH dataset saved in .h5 format
- output_images(runtime directory) - the outputs of the image predictions (check this for results!)



### Build Instructions

This project is built using [Anaconda](https://www.anaconda.com/products/individual)

There are two ways to build the project.

1) Using the class specified cv_project.yml 

```
conda env create -f cv_proj.yml
conda activate cv_proj
```

2) Using the mini conda environment I set up specifically to my own machine with updated CUDA drivers.  

```
conda env create -f cnn_build.yml
conda activate cnn_build
```



### Run Instructions

Once the virtual environments are activated, the entire pipeline can be run by executing

```
python run.py
```

### Sample Results

![alt text](./readme_images/predict_0.png)

![alt text](./readme_images/predict_1.png)

![alt text](./readme_images/predict_2.png)

### Future Work

Dockerize!