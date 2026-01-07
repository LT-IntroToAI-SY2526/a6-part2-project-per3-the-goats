"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Enggy Puma
- Victoria Serrano
- 
- 

Dataset: [Smartphone market prices]
Predicting: [Resale price of phones]
Features: [brand, condition, age]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'smartphone_prices.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    data = pd.read_csv(filename, sep="\t")

    print("=== Smartphone Market Prices ===")
    print(f"\First 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print(f"\nBasic statistics:")
    print(data.describe())

    print(f"\nColumn names: {list(data.columns)}")

    return data



def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Smartphone Features vs Price', fontsize=16, fontweight='bold')
    
   
    axes[0, 0].scatter(data['Brand'], data['Price'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Brand (0=Apple, 1=Samsung, 2=Google)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Brand vs Price')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(data['Condition'], data['Price'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Condition (0=Poor, 1=Fair, 2=Good, 3=New)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Condition vs Price')
    axes[0, 1].grid(True, alpha=0.3)
   

    axes[1, 0].scatter(data['Age'], data['Price'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Age vs Price')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].text(0.5, 0.5, 'Space for additional feautures',
                   ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature plots saved as 'feature_plots.png'")
    
    plt.show()
    


def prepare_feautures(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    feauture_columns = ['Brand', 'Condition', 'Age']
    X = data[feauture_columns]
    y = data['Price']

    print(f"\n=== Feauture Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    
    return X, y


def split_data(X, y):

    X_train = X.iloc[:15]
    X_test = X.iloc[15:]   
    y_train = y.iloc[:15]
    y_test = y.iloc[15:]

    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    print(f"\nEquation:")
    equation = f"Price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    
    return model


def evaluate_model(model, X_test, y_test, feauture_names):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feauture_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    
    return predictions
    
def compare_predictions(y_test, predictions, num_examples=10):

    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)

    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100

        print(f"${actual:>12.2f}  ${predicted:>13.2f}  ${error:>10.2f}  {pct_error:>6.2f}%")


def make_prediction(model, brand, condition, age):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    smartphone_feautures = pd.DataFrame([[brand, condition, age]],
                                        columns=['Brand', 'Condition', 'Age'])
    predicted_price = model.predict(smartphone_feautures)[0] 

    brand_name = ['Apple', 'Samsung', 'Google'][brand]
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    print(f"\n=== New Prediction ===")
    print(f"Predicted smartphone price: ${predicted_price:,.2f}")

    return predicted_price


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare feautures
    X, y = prepare_feautures

    X_train, X_test, y_train, y_test = split_data(X, y)
    
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

