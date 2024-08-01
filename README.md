Neural Network from Scratch for Boston Housing Dataset
This project demonstrates building a neural network from scratch to predict housing prices using the Boston Housing Dataset. The notebook includes steps for data preprocessing, visualization, model building, training, and evaluation.

Table of Contents
Introduction
Dataset
Requirements
Usage
Project Structure
Results
Contributing

Introduction
This project involves creating a neural network from scratch using Python and numpy to predict the median value of owner-occupied homes in the Boston area. The dataset is preprocessed, and a custom neural network class is implemented and trained.

Dataset
The dataset used is the Boston Housing Dataset, which contains information about various factors influencing housing prices in Boston. It includes features such as crime rate, average number of rooms per dwelling, and more.

Requirements
To run the code in this notebook, you need the following libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
You can install these using pip:


pip install numpy pandas matplotlib seaborn scikit-learn
Usage
To use this notebook, follow these steps:


Open the Jupyter Notebook:
jupyter notebook DL_proj_stoch_.ipynb
Run the cells in the notebook to execute the code step by step.
Project Structure
The project consists of the following main sections:

Importing Libraries: Importing necessary libraries for the project.
Load and Preprocess the Dataset: Loading the Boston Housing Dataset and performing initial preprocessing.
Data Visualization: Visualizing the dataset using correlation matrices and scatter plots.
Standardizing Data and Splitting: Standardizing the dataset and splitting it into training and testing sets.
Building the Neural Network: Implementing the neural network from scratch using numpy.
Training the Model: Training the neural network using stochastic, batch and mini batch gradient descent.
Evaluating the Model: Plotting the learning curve to evaluate the model's performance.
Results
The model's performance is evaluated using the mean squared error on the training and testing sets. The learning curve is plotted to visualize the training process and the model's convergence.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
