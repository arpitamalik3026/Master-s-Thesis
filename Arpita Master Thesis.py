import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = r"C:\Users\NiravG\Documents\Appy Thesis.xlsx"
df = pd.read_excel(file_path)

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df['Month'] = df['Date'].dt.month_name()

# Aggregate data at the Month, Category, and Item levels
aggregated_data = df.groupby(['Month', 'Category', 'Item'], as_index=False).agg({
    'Quantity': 'sum',
    'Sales': 'sum'
})

# Encode Month, Category, and Item as numerical features
aggregated_data['Month_Num'] = aggregated_data['Month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
})
aggregated_data['Category_Code'] = aggregated_data['Category'].astype('category').cat.codes
aggregated_data['Item_Code'] = aggregated_data['Item'].astype('category').cat.codes

# Define features and target variable
X = aggregated_data[['Month_Num', 'Category_Code', 'Item_Code', 'Quantity']]
y = (aggregated_data['Sales'] > aggregated_data['Sales'].median()).astype(int)  # Binarize sales as high/low

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function for hyperparameter tuning and evaluation
def evaluate_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return best_model

# Hyperparameter grids for models

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Evaluate XGBoost
print("\nXGBoost Results:")
best_xgb = evaluate_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), param_grid_xgb)
