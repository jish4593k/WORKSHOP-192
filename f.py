
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog

def load_data_gui():
    """Load data interactively using Tkinter GUI."""
    root = Tk()
    root.title("Data Loading GUI")

    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if file_path:
        data = pd.read_csv(file_path)
        root.destroy()
        return data
    else:
        root.destroy()
        return None

def preprocess_data(df):
    """Preprocess data and perform regression analysis."""
    # Convert mass and radius columns to numeric
    df['Mass'] = pd.to_numeric(df['Mass'], errors='coerce')
    df['Radius'] = pd.to_numeric(df['Radius'], errors='coerce')

    # Drop rows with missing values
    df.dropna(subset=['Mass', 'Radius'], inplace=True)

    # Convert star masses to solar masses in kg
    df['Solar Mass (kg)'] = df['Mass'] * 1.989e+30

    # Convert star radii to meters
    df['Solar Radius (m)'] = df['Radius'] * 6.957e+8

    # Calculate star gravities
    df['Solar Gravity'] = (df['Solar Mass (kg)'] * 5.972e+24) / \
                          (df['Solar Radius (m)']**2 * 6371000**2 * 6.674e-11)

    return df

def perform_regression(X, y):
    """Perform regression using TensorFlow and Keras."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

   
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

  
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

  
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions.flatten())
    plt.title('Actual vs. Predicted Solar Gravity')
    plt.xlabel('Actual Solar Gravity')
    plt.ylabel('Predicted Solar Gravity')
    plt.show()

def main():
    print('Welcome to Advanced Data Analysis')

    data = load_data_gui()

    if data is not None:
        
        processed_data = preprocess_data(data)

        X = processed_data[['Solar Mass (kg)', 'Solar Radius (m)']]
        y = processed_data['Solar Gravity']

        perform_regression(X, y)

if __name__ == "__main__":
    main()
