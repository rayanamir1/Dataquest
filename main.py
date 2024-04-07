import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Changed from RandomForestRegressor to RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Define MAE function
def get_mae(max_leaf_nodes, train_X, val_X, train_Y, val_Y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_Y)
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_Y, predictions)
    return mae

# Define metrics function for the Random Forest Classifier
def get_metrics(n_estimators, max_depth, train_X, val_X, train_Y, val_Y):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    model.fit(train_X, train_Y)
    predictions = model.predict(val_X)
    accuracy = accuracy_score(val_Y, predictions)
    precision = precision_score(val_Y, predictions)
    f1 = f1_score(val_Y, predictions)
    return [accuracy, precision, f1]

# Load the training data
train_data = pd.read_excel("data.csv.xlsx", sheet_name="train")
train_data = train_data.dropna()

# Process categorical data: 'category'
cate = {
    "entertainment": 0,
    "food_dining": 1,
    "gas_transport": 2,
    "grocery_net": 3,
    "grocery_pos": 4,
    "health_fitness": 5,
    "home": 6,
    "kids_pets": 7,
    "misc_net": 8,
    "misc_pos": 9,
    "personal_care": 10,
    "shopping_net": 11,
    "shopping_pos": 12,
    "travel": 13
}
train_data['category'] = train_data['category'].map(cate)

# Process date columns
train_data['dateOfBirth'] = pd.to_datetime(train_data['dateOfBirth']).dt.year
train_data['transDate'] = train_data['transDate'].astype(str).str[-4:]

# Factorize the 'job' column
train_data['job_code'], _ = pd.factorize(train_data['job'])

# Define the features for the model
features = ["category", "dateOfBirth", "amount", "job_code", "transDate", "cityPop", "latitude", "longitude", "merchLatitude", "merchLongitude"]

# Make sure all features exist in the dataframe and are not strings
features = [f for f in features if f in train_data.columns and train_data[f].dtype != 'object']

X = train_data[features]
y = train_data['isFraud']

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Train the model and compute MAE for different max_leaf_nodes
for max_leaf_nodes in [5, 50, 500]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean absolute error: %f" % (max_leaf_nodes, my_mae))

# Train the RandomForestClassifier and evaluate metrics
print("Data Processing...")
for n_estimators in [10, 100]:
    for max_depth in [5, 10, None]:  # None means no maximum depth limit
        metrics = get_metrics(n_estimators, max_depth, train_X, val_X, train_y, val_y)
        print(f"Estimators: {n_estimators}, Max Depth: {max_depth}\t\t Accuracy: {metrics[0]:.2f}, Precision: {metrics[1]:.2f}, F1 Score: {metrics[2]:.2f}")

# Optional visualization
#scatter_matrix(X, alpha=0.2, figsize=(10, 10), diagonal='kde')
#plt.show()

# ... [previous code unchanged] ...

# Train the best model on the entire training dataset
# For simplicity, let's assume the best model is with max_leaf_nodes=50 based on previous results
best_model = DecisionTreeRegressor(max_leaf_nodes=50, random_state=0)
best_model.fit(X, y)

# Load the test data
test_data = pd.read_excel("data.csv.xlsx", sheet_name="test")

# Repeat the preprocessing steps for the test data
test_data['category'] = test_data['category'].map(cate)
test_data['dateOfBirth'] = pd.to_datetime(test_data['dateOfBirth']).dt.year
test_data['transDate'] = test_data['transDate'].astype(str).str[-4:]
test_data['job_code'], _ = pd.factorize(test_data['job'])

# Prepare the test feature matrix
X_test = test_data[features]

# Predict on the test data
test_data['predicted_isFraud'] = best_model.predict(X_test)

# Convert predictions to 0 or 1
test_data['predicted_isFraud'] = (test_data['predicted_isFraud'] > 0.5).astype(int)

# Save the predictions to a new Excel file or the same file in a different sheet
test_data.to_excel("predicted_isFraud.xlsx", index=False)

print("Predictions saved to predicted_isFraud.xlsx.")
