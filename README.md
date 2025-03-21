# Movie Rating Prediction

This project aims to predict movie ratings based on different attributes using machine learning.

## Approach:
1. **Data Preprocessing**:
   - Missing values in the dataset were handled using mean or mode.
   - Categorical variables such as Director and Genre were encoded using OneHotEncoder.
   
2. **Feature Engineering**:
   - We created new features like `Director_Success_Rate` (average rating of a director's movies) and `Avg_Rating_Similar_Movies` (average rating of movies in the same genre).

3. **Model**:
   - We used a RandomForestRegressor to predict movie ratings.

4. **Evaluation**:
   - Model performance was evaluated using **MAE**, **MSE**, and **R² Score**.

## Requirements:
- Python 3.x
- pandas
- numpy
- scikit-learn

## Installation:
1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
