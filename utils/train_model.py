"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')

train = train[(train['Commodities'] == 'APPLE GOLDEN DELICIOUS')]

y_train = train['avg_price_per_kg']
X_train = train[['Total_Qty_Sold','Stock_On_Hand']]

# Fit model
mlr_model = LinearRegression(normalize=True)
print ("Training Model...")
mlr_model.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/mlr_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(mlr_model, open(save_path,'wb'))
