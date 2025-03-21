# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset from the specified path with a different encoding
file_path = r'C:\Users\saira\Downloads\Movie-Rating-Prediction\data\IMDb Movies India.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Using ISO-8859-1 encoding to handle non-UTF-8 characters

# Print the first few rows of the dataset to check if it's loaded correctly
print("Dataset Loaded Successfully")
print(data.head())

# Step 2: Data Preprocessing
# Handle missing values (example: fill with mean or mode)
data['Rating'] = data['Rating'].fillna(data['Rating'].mean())  # Replace missing ratings with mean
data['Director'] = data['Director'].fillna(data['Director'].mode()[0])  # Replace missing directors with the mode
data['Genre'] = data['Genre'].fillna(data['Genre'].mode()[0])  # Replace missing genres with the mode

# Feature Engineering: Director Success Rate (average rating of director's movies)
director_success_rate = data.groupby('Director')['Rating'].mean()  # Calculate average rating for each director
data['Director_Success_Rate'] = data['Director'].map(director_success_rate)  # Map this to the original data

# Feature Engineering: Average Rating of Similar Movies (movies with the same genre)
genre_avg_rating = data.groupby('Genre')['Rating'].mean()  # Calculate average rating for each genre
data['Avg_Rating_Similar_Movies'] = data['Genre'].map(genre_avg_rating)  # Map this to the original data

# Print the data to check if the columns are created correctly
print("Data After Feature Engineering:")
print(data[['Director', 'Genre', 'Director_Success_Rate', 'Avg_Rating_Similar_Movies']].head())

# Step 3: Drop unnecessary columns (only drop if the column exists)
columns_to_drop = ['Title', 'Year']
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]  # Check if columns exist
data.drop(existing_columns_to_drop, axis=1, inplace=True)  # Drop columns

# Step 4: Encode categorical variables (e.g., Genre, Director)
categorical_features = ['Director', 'Genre']
numeric_features = ['Director_Success_Rate', 'Avg_Rating_Similar_Movies']

# Define the preprocessing steps for numerical and categorical data
numerical_transformer = StandardScaler()  # Scale numerical features
categorical_transformer = OneHotEncoder(drop='first')  # One-hot encode categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Split the data into features and target variable
X = data.drop('Rating', axis=1)  # Features (all columns except 'Rating')
y = data['Rating']  # Target variable (the 'Rating' column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of training and testing sets
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Step 6: Build the Model Pipeline
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random forest regressor model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])  # Create a pipeline with preprocessing and model

# Step 7: Train the model
pipeline.fit(X_train, y_train)  # Fit the model to the training data

# Step 8: Evaluate the model
y_pred = pipeline.predict(X_test)  # Make predictions on the test set

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared Score

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
