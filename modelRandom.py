import pandas as pd
from RandomForest import RandomForest

def load_data(file_path):
    # Load Excel data into a DataFrame
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    # No preprocessing required for this example
    return df

def main():
    # Path to your Excel file
    file_path = 'C:/New Volume/project 2k24/project2k24/DataAnalysis_MOdel_traiining/potato/potato_dataset.xlsx'

    # Load and preprocess the data
    df = load_data(file_path)
    df_processed = preprocess_data(df)

    # Assuming 'target_column' is the target variable column
    X = df_processed.iloc[:, 1:-1].values  # Features
    y = df_processed.iloc[:, -1].values    # Labels

    # Initialize and train the RandomForest model
    rf_model = RandomForest(n_trees=50, max_depth=10, min_samples_split=2)
    rf_model.fit(X, y)

    # Make predictions (using the same dataset for demonstration)
    y_pred = rf_model.predict(X)

    # Evaluate model performance (using accuracy for demonstration)
    accuracy = (y_pred == y).mean()
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
