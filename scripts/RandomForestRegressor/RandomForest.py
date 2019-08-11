import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
# Import the train_test_split function
from sklearn.model_selection import train_test_split

# get absolute error using max leaf nodes
def get_mae( train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Path of the file to read
iowa_file_path = '../../data/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# split our data to trainning and validation models
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

error = get_mae(train_X, val_X, train_y, val_y)

print("error minimum than previous algorithms")
print(error)
