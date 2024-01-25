import pandas as pd
import pickle
import os

import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
from statsmodels.iolib.smpickle import load_pickle

# Levels observed in categorical fields, this is necessary for creating the same dummies in prediction as training
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
locations = ["asia", "germany", "japan"]
gender = ["Male", "Female"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
parameters = [f"x{i}" for i in range(0, 100)]

# Loads the segmentation model from a saved pickle object
def load_model():
    return sm.load("objs/ad_purchase_model.pickle")

# Load necessary objects that were used in training to maintain consistency
result = load_model()
with open("objs/variables.pickle", "rb") as input:
    variables = pickle.load(input)
with open("objs/imputer.pkl", "rb") as input:
    imputer = pickle.load(input)
with open("objs/std_scaler.pkl", "rb") as input:
    std_scaler = pickle.load(input)

# Complete pipeline to transform incoming data to the state necessary for working with our model
def prepare_model_input(data):
    # Format string features for money/percentage to be floats
    data = format_input(data)

    # Impute missing fields and scale data, we will use the same fitted objects from training
    data_imputed = pd.DataFrame(imputer.transform(data.drop(columns=['x5', 'x31',  'x81' ,'x82'])), columns=data.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
    data_imputed_std = pd.DataFrame(std_scaler.transform(data_imputed), columns=data_imputed.columns)

    # Create one-hot encodings of data
    data_imputed_std = encode_input(data, data_imputed_std)

    # Subset data by features chosen during training's feature selection
    data_prediction = data_imputed_std[variables]

    return data_prediction

# Formats the x12 and x63 fields into floats from strings
def format_input(data_val):
    #1. Fixing the money and percents#
    data_val['x12'] = data_val['x12'].str.replace('$','')
    data_val['x12'] = data_val['x12'].str.replace(',','')
    data_val['x12'] = data_val['x12'].str.replace(')','')
    data_val['x12'] = data_val['x12'].str.replace('(','-')
    data_val['x12'] = data_val['x12'].astype(float)
    data_val['x63'] = data_val['x63'].str.replace('%','')
    data_val['x63'] = data_val['x63'].astype(float)

    return data_val

# Create dummy encodings of our categorical fields
def encode_input(data, data_imputed_std):
    # Creating dummy variables for a day of the week categorical variable
    dumb5 = pd.get_dummies(data['x5'].astype(pd.CategoricalDtype(categories=days)), prefix='x5', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb5], axis=1, sort=False)

    # Creating dummy variables for a location categorical variable
    dumb31 = pd.get_dummies(data['x31'].astype(pd.CategoricalDtype(categories=locations)), prefix='x31', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb31], axis=1, sort=False)

    # Creating dummy variables for a month of the year categorical variable
    dumb81 = pd.get_dummies(data['x81'].astype(pd.CategoricalDtype(categories=months)), prefix='x81', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb81], axis=1, sort=False)

    # Creating dummy variables for a gender categorical variable
    dumb82 = pd.get_dummies(data['x82'].astype(pd.CategoricalDtype(categories=gender)), prefix='x82', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb82], axis=1, sort=False)

    del dumb5, dumb31, dumb81, dumb82

    return data_imputed_std

# Validate the correctness of the user's input
def validate_input(data):
    # Validate that the dataframe has the correct number of columns
    if len(data.columns) != 100:
        return f"Number of parameters is {len(data.columns)}, not 100 parameters", 410
    
    # Validate input for different categorical fields that were trained using one-hot encodings
    x5_unique = data["x5"].unique()
    for val in x5_unique:
        if val not in days:
            return f"Invalid category of {val} appeared in feature x5, \
                    valid features include [monday, tuesday, wednesday, thursday, friday, saturday, sunday]", 420
    x31_unique = data["x31"].unique()
    for val in x31_unique:
        if val not in locations:
            return f"Invalid category of {val} appeared in feature x31, \
                    valid features include [asia, japan, america, germany]", 421
    x81_unique = data["x81"].unique()
    for val in x81_unique:
        if val not in months:
            return f"Invalid category of {val} appeared in feature x81, \
                    valid features include [January, February, ..., December]", 422
    x82_unique = data["x82"].unique()
    for val in x82_unique:
        if val not in gender:
            return f"Invalid category of {val} appeared in feature x82, \
                    valid features include [Male, Female]", 423

    # Validate that we have the necessary objects from training
    if not os.path.isfile("objs/ad_purchase_model.pickle"):
        return "Missing saved model object, please add \"ad_purchase_model.pickle\" object to working directory", 510
    if not os.path.isfile("objs/variables.pickle"):
        return "Missing saved selected features object from training, please add \"features.pickle\" object to working directory", 511
    if not os.path.isfile("objs/imputer.pkl"):
        return "Missing saved fitted imputer object from training, please add \"imputer.pickle\" object to working directory", 512
    if not os.path.isfile("objs/std_scaler.pkl"):
        return "Missing saved fitted std scaler object from training, please add \"std_scaler.pickle\" object to working directory", 513

    return "Valid", 200
