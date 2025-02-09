import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load the dataset
file_path = r"C:\Users\NiravG\Documents\Appy Thesis.xlsx"
df = pd.read_excel(file_path)

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df['Month'] = df['Date'].dt.month_name()

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# Remove duplicate rows
df_cleaned = df.drop_duplicates()

# Extract month and weekday to analyze seasonality
df_cleaned["Month"] = df_cleaned["Date"].dt.month_name()
df_cleaned["Weekday"] = df_cleaned["Date"].dt.day_name()

# Aggregate sales by month
monthly_sales = df_cleaned.groupby("Month")["Sales"].sum()

# Order the months correctly
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
monthly_sales = monthly_sales.reindex(month_order)

# Aggregate sales by category and month
category_monthly_sales = df_cleaned.pivot_table(values="Sales", index="Month", columns="Category", aggfunc="sum")
category_monthly_sales = category_monthly_sales.reindex(month_order)

# Aggregate sales by weekday
weekday_sales = df_cleaned.groupby("Weekday")["Sales"].sum()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_sales = weekday_sales.reindex(weekday_order)

# 1. Monthly Sales Trend Analysis
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind="bar", color="royalblue")
plt.xlabel("Month")
plt.ylabel("Total Sales ($)")
plt.title("Monthly Sales Trend")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 2. Category Performance Over Time
category_monthly_sales.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.xlabel("Month")
plt.ylabel("Total Sales ($)")
plt.title("Category-Wise Sales Performance Over Months")
plt.xticks(rotation=45)
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis='y')
plt.show()

# 3. Price Sensitivity Analysis
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned["Rate"], df_cleaned["Sales"], alpha=0.5, color='purple')
plt.xlabel("Price per Item ($)")
plt.ylabel("Total Sales ($)")
plt.title("Price Sensitivity Analysis: Rate vs Sales")
plt.grid(True)
plt.show()

# 4. Outlier Detection in Sales
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned["Sales"], color="red")
plt.xlabel("Sales ($)")
plt.title("Outlier Detection in Sales Data")
plt.grid(True)
plt.show()

# 5. Sales Performance by Weekday
plt.figure(figsize=(12, 6))
weekday_sales.plot(kind="bar", color="green")
plt.xlabel("Day of the Week")
plt.ylabel("Total Sales ($)")
plt.title("Sales Performance by Weekday")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

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

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Low Sales', 'High Sales'], yticklabels=['Low Sales', 'High Sales'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

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
