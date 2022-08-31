import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import Series, DataFrame
from patsy import dmatrices
import sklearn.ensemble as ske

#If after installing the requirements.txt you run into a ModuleNotFoundError, just install the packages manually with pip

class Ingestion:
    #Read the data for training
    global df
    df = pd.read_csv("../data/train.csv")

class FeatureEngineering:
    # Drop ticket and Cabin to preserve integrity of the dataset
    df = df.drop(['Ticket','Cabin'], axis=1)
    # Remove NaN values
    df = df.dropna()

class Training:
    # model formula
    # here the ~ sign is an = sign, and the features of our dataset
    # are written as a formula to predict survived. The C() lets our 
    # regression know that those variables are categorical.
    # Ref: http://patsy.readthedocs.org/en/latest/formulas.html
    formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)' 
    # create a results dictionary to hold our regression results for easy analysis later        
    results = {} 

    # create a regression friendly dataframe using patsy's dmatrices function
    y,x = dmatrices(formula, data=df, return_type='dataframe')

    # instantiate our model
    model = sm.Logit(y,x)

    # fit our model to the training data
    res = model.fit()

    # save the result for outputing predictions later
    results['Logit'] = [res, formula]
    res.summary()

class Training2:
    #Alternative model
    # Create an acceptable formula for our machine learning algorithm
    formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
    # Create the random forest model and fit the model to our training data
    y, x = dmatrices(formula_ml, data=df, return_type='dataframe')
    # RandomForestClassifier expects a 1 demensional NumPy array, so we convert
    y = np.asarray(y).ravel()
    #instantiate and fit our model
    results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)