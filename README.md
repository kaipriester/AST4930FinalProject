# AST4930 Final Project
## CAMS Meteor Detection with Various Fundamental Supervised ML Algorithms

    Kai Priester

    University of Florida Fall 2021

**Purpose**

Researchers at FDL developed an elegant pipeline which continuously takes in new meteor data and improves the classification model. Each light streak’s properties are recorded to the data store. The only features extracted and used in the model are latitude, longitude, velocity, height, and magnetic flux. The training and test data sets from past years also have a binary classification feature. 1 indicates a true meteor data point and 0 indicates a false meteor data point. The type of model used is a complex neural network. It is a long short-term memory (LSTM) binary classifier with 2 stacked bidirectional LSTM layers with 512 nodes and 4 dense layers. Every layer has a rectified linear activation function (RELU), except for the last layer, which has sigmoid activation appropriate for binary classification. The loss function used is a binary cross-entropy, with an Adam optimizer. 

Such a dense neural network is computationally heavy to continuously train. It is reported that the models have almost perfect classification accuracy. However, all this training requires NVIDIA TITAN GPUs in order to compute in a reasonable amount of time. With FDL’s focus on citizen science and approachability, I wondered why they chose such an esoteric model method. 

Fundamental supervised machine learning methods such as k-nearest neighbors (KNN), decision trees (DT), and support-vector machines (SVM) are relatively straightforward. After observing the CAMS data set, I thought it would be feasible to build these types of models. Using one of these algorithms would make the machine learning module less of a black box and in turn incite greater interest in machine learning. Therefore, the purpose of my project is to build the ideal KNN, DT, and SVM meteor classification models using the CAMS datastore. 
I plan to inspect the time complexity, optimal hyperparameters, model accuracy, and overall behavior of each model to gain insights on the nature of the dataset. Overall, I would like to uncover reasons the FDL researchers do not use one of the fundamental model methods in their detection pipeline. 

**How to run experiment**

1. Experiment is based on this project, so please read this to set-up environment https://github.com/sidgan/CAMS-NCF

2. Download data from here and place in corresponding folders https://drive.google.com/drive/folders/1qKzWYR3U6pEvw5HzBC7v4hkWi47src7M

3. Execute: `python3 script.py`