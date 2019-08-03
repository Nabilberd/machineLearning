import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
# Import the train_test_split function
from sklearn.model_selection import train_test_split

# Path of the file to read
iowa_file_path = '../data/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# split our data to trainning and validation models
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model
iowa_model.fit(train_X, train_y)

# Get predicted prices on validation data
val_predictions = iowa_model.predict(val_X)
print("predictions", val_predictions)

# Test difference between real price and predicted
val_predictions = iowa_model.predict(val_X.head())
print("predictions", val_predictions, y.head().toList())

# Print accuracy of the model
accuracy = accuracy_score(val_y, val_predictions)
print("accuracy", accuracy)

print("error", mean_absolute_error(val_predictions, val_y))

