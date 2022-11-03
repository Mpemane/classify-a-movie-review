# Project title: Classify a movie review
## Project Description
Use Machine Learning to build a RNN model which you can use to classify movie reviews as either negative or positive, which is a sentiment analysis project.
 ## Table of Content
 ### 1. Introduction
 ### 2. Technologies
 ### 3. Layout of the of the Jupyter Notebook
 ### 4. Conclusion

 ### Introduction
 The objective of this project is to use two different neural networks in Keras to classify a book review as either positive
or negative, and report on which network type worked better
 ### Technologies
 The Technologies employed in this project are at follows:
 ### Libraries used:
 import numpy as np
 import pandas as pd
 #### Visualisazation libraries:
 import seaborn as sns
 import matplotlib.pyplot as plt
 %matplotlib inline
 #### sklearn, tensor-flow and keras libraries used in the project:
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Sequential
from keras.layers import Dropout, LSTM
from keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
 
 ### Layout of the Jupyter NoteBook
Get the dataset
Preprocess the Data, 
Build the Model, 
Train the model, 
Test the Model and
Prediction from unseen data
 ### Conclusion 
 The model worked well, as it was able to predict the Sentimental on unseen data, However the both models performed with an accuracy of 50%, which suggest that thses model are not best suited for the dataset.
