import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn import model_selection
# Import the train_test_split function
from sklearn.model_selection import train_test_split

# Path of the file to read
iowa_file_path = '../../data/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

train_X, val_X, train_y, val_y = model_selection.train_test_split(X,y,random_state=1)

model = LogisticRegression()
modeleRegression = model.fit(train_X,train_y)

predicted_classes = modeleRegression.predict(val_X)

# Test difference between real price and predicted
val_predictions = modeleRegression.predict(val_X.head())
print("predictions", val_predictions, y.head())

accuracy = accuracy_score(val_y,predicted_classes)

print(accuracy)

print("error", mean_absolute_error(predicted_classes, val_y))

plt.figure()
plt.plot(train_X, train_y, linestyle='', marker='x', color='r', label='Training data')
plt.show()