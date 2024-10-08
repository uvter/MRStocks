# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, flash
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a strong secret key

# Set random seed for reproducibility
np.random.seed(42)  # Set seed for NumPy random operations

def fetch_and_analyze_stock(stock_ticker, seed_values):
    # Calculate the date range for the last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months = 180 days
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    if stock_data is None or stock_data.empty:
        return [f"Error: Could not fetch data for {stock_ticker}. Please try again."]

    stock_data = stock_data[['Adj Close']].copy()
    stock_data['Prediction'] = stock_data['Adj Close'].shift(-30)
    X = np.array(stock_data.drop(['Prediction'], axis=1))[:-30]
    y = np.array(stock_data['Prediction'])[:-30]

    if len(X) < 30:
        return [f"Error: Not enough data to analyze {stock_ticker}."]

    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Initialize a summary table
    summary_table = []
    plots = []  # Initialize a list for plots

    # Iterate over each seed value
    for seed in seed_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        seed_results = {'seed': seed}  # Ensure seed is included in results

        # Store results for each model
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate performance metrics
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))  # RMSE
            r2 = model.score(X_test, y_test)  # R²

            # Append results to the summary table
            seed_results[model_name] = {'RMSE': rmse, 'R²': r2}

            # Plotting actual vs predicted prices
            plt.figure(figsize=(10, 5))

            # Plot actual prices (the last 30 prices)
            plt.plot(stock_data.index[-30:], stock_data['Adj Close'].iloc[-30:], label='Actual Prices')

            # Ensure predictions are plotted against the correct dates
            predicted_dates = stock_data.index[-len(predictions):]  # Adjust for the length of predictions
            plt.plot(predicted_dates, predictions, label=f'{model_name} Predictions', linestyle='--')  # Use model_name as label

            plt.title(f'{stock_ticker} Price Prediction with Seed {seed} - {model_name}')  # Include model name in title
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()  # Show the legend
            plot_filename = f'static/{stock_ticker}_prediction_seed_{seed}_{model_name}.png'  # Include model name in the filename
            plt.savefig(plot_filename)  # Save the plot
            plt.close()  # Close the figure to free memory
            plots.append(plot_filename)  # Add the filename to the list

        summary_table.append(seed_results)

    return summary_table, plots  # Return both the summary table and plots


@app.route('/', methods=['GET', 'POST'])
def home():
    ml_results = None  # Initialize result variable
    plots = []  # Initialize plots variable
    if request.method == 'POST':
        stock_ticker = request.form['ticker'].upper()
        
        # Use .get() method to avoid KeyError and set a default if not provided
        seeds_input = request.form.get('seeds', '42')  # Default to '42' if no seeds provided
        seed_values = seeds_input.split(',')  # Get seeds from input as a list of strings
        
        # Validate and convert to integers
        try:
            seed_values = [int(seed.strip()) for seed in seed_values if seed.strip().isdigit()]  # Convert to integers
        except ValueError:
            flash("Error: Please enter valid integer seed values.")  # Flash an error message
            return render_template('home.html')  # Render the home page again

        # Fetch and analyze stock for each seed and store results
        ml_results, plots = fetch_and_analyze_stock(stock_ticker, seed_values)  # Pass seeds to the analysis
        
        # Debugging line to see the structure of the results
        print(ml_results)

    return render_template('home.html', ml_results=ml_results, plots=plots)  # Render home page with results


# Run the Flask app
if __name__ == '__main__':
    # Create the static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
