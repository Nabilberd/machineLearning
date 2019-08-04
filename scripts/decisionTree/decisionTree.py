import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
# Import the train_test_split function
from sklearn.model_selection import train_test_split

# get absolute error using max leaf nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
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

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model
iowa_model.fit(train_X, train_y)

# Get predicted prices on validation data
val_predictions = iowa_model.predict(val_X)
print("predictions", val_predictions)

# Print accuracy of the model
accuracy = accuracy_score(val_y, val_predictions)
print("accuracy", accuracy)

print("error", mean_absolute_error(val_predictions, val_y))

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

print("best_tree_size", round(scores.get(min(scores, key=scores.get))))

