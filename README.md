# Spam Email Classifier

This project is about creating a machine learning model to classify spam emails. The model is trained on a dataset which contains examples of both spam and non-spam ("ham") emails.

## Project Structure

The main Python script of this project is `ML.py` which does the following:

- Imports necessary libraries and the spam dataset
- Splits the dataset into training and test sets
- Trains a Support Vector Machine (SVM) model with grid search to find the best parameters
- Prints out the best parameters found
- Tests the trained model on the test set and prints out the accuracy

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.7 or later.
* You have installed the following Python libraries: pandas, sklearn.
* You have a basic understanding of Machine Learning concepts including training/testing datasets and classification algorithms.

## Using Spam Email Classifier

To use Spam Email Classifier, follow these steps:

* Clone this repository to your local machine
* Navigate to the directory of the cloned repo in your terminal
* Run the following command:

```bash
python ML.py

