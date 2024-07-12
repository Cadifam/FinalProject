## Loan status prediction

Predict the loan to be approved or rejected for an applicant.

## About dataset

In this Loan Status Prediction dataset, we have the data of applicants who previously applied for the loan based on the property which is a Property Loan. The bank will decide whether to give a loan to the applicant based on some factors such as Applicant Income, Loan Amount, previous Credit History, Co-applicant Income, etc… Our goal is to build a Machine Learning Model to predict the loan to be approved or to be rejected for an applicant.

## About the loan_data.csv file:

    -Loan_ID: A unique loan ID.
    -Gender: Either male or female.
    -Married: Weather Married(yes) or Not Marttied(No).
    -Dependents: Number of persons depending on the client.
    -Education: Applicant Education(Graduate or Undergraduate).
    -Self_Employed: Self-employed (Yes/No).
    -ApplicantIncome: Applicant income.
    -CoapplicantIncome: Co-applicant income.
    -LoanAmount: Loan amount in thousands.
    -Loan_Amount_Term: Terms of the loan in months.
    -Credit_History: Credit history meets guidelines.
    -Property_Area: Applicants are living either Urban, Semi-Urban or Rural.
    -Loan_Status: Loan approved (Y/N).

## Goal:

In this project, we are going to classify an individual whether he/she can get the loan amount based on his/her Income, Education, Working Experience, Loan taken previously, and many more factors. Let’s get more into it by looking at the data.

## The project is divided into three (3) parts/stages:

    - Load data for preprocesing and save preprocessed data for (EDA)
    - Perform Exploratory Data Analysis on the preprocessed data and save model
    - Load saved model and test the model

## Environment Configuration (Installing virtual Env):

    -pip install pipenv

    Using github;
        -create a repo with your github account
        -clone the repo on your local directory
        -change directory from your local repo to the cloned github repo

    Installing Packages
        pipenv install:
        -jupyter notebook
        -pandas
        -numpy
        -matplotlib
        -seaborn
        -scikit-learn
        -pyarrow

## Starting/Stopping Virtual Env

    Starting Notebook
        -pipenv shell
        -jupyter-notebook

    Stoping Notebook
        -Ctrl+c

    Deactiving Virtual Env
        -exit

## Load data reviewing of data:

    Import libraries
        -import numpy as np
        -import pandas as pd

    Load and perform overview of dataset
        -df.head()
        -df.tail().T
        -df.info()
        -df.shape()
        -df.dtypes
        -df.isnull().sum()
        -unique instances() etc

## Data preprocessing:

    Import libraries
        -import numpy as np
        -import pandas as pd

    Data Preprocessing
        -Normalize the column names to lower case
        -Drop the ID column
        -Remove the (+) sign on the Dependants column
        -Fill the NaN in the (Dependants, Credit_History, Loan_Amount, Gender, Self_Employed) columns
        -Replace categorical column(Loan-Status) with integers
        -Save cleaned dataset

## Exploratory Data Analysis (EDA):

    Import libraries
        -import numpy as np
        -import pandas as pd
        -import matplotlib.pyplot as plt
        -import seaborn as sns
        -from sklearn.model_selection import train_test_split
        -from sklearn.feature_extraction import DictVectorizer
        -from sklearn.linear_model import LogisticRegression
        -from sklearn.metrics import accuracy_score
        -import pickle

## Target Variable Analysis:

    -Load the cleaned loan dataset
    -Perform a target variable analysis
    -Build a Validation Framework
    -Divide the dataset into three (3)
        -Training dataset 60%
        -Validation dataset 20%
        -Testing dataset 20%

## Feature Engineering:

    -Seperate the dataset into numerical attributes and categorical attributes
    -perform the one-hot encoding
    -convert the dataframe into dict
    -DictVectorizer
    -(fit) the train_dict

## Train The Model:

    -LogisticRegression
    -compaire predicted truth vrs ground truth

The predictions of the model: a two-column matrix. The first column contains the probability that the target is zero (the application will be approved). The second column contains the opposite probability (the target is one, and the application will be rejected).
The output of the (probabilities) is often called soft predictions. These tell us the probability of rejection as a number between zero and one. It’s up to us to decide how to interpret this number and how to use it.

## Saving The Model:

    -import pickle
    -specifyging where to save the file
    -save the model

## Testing the Model:

    Load libraries
        -import numpy as np
        -import pandas as pd
        -import pickle

    -load the saved model
    -Load applicant Data to predict status (Approve/Reject) of application

    Models's verdict:

        -if prediction >= 0.5, applicant is in a good financial standing to pay back loan; therefore "Approve"
        -if prediction <= 0.5, applicant is not in a good financial standing to pay back loan; therefore "Reject"
