# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import cohere
from flask import Flask, render_template, request, session, redirect, url_for
import threading

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed to use session for storing the API key

def fetch_and_analyze_stock(stock_ticker, co):
    result = {
        "data": None,
        "error": None
    }
    
    def fetch_data():
        nonlocal result
        stock_data = yf.download(stock_ticker, start="2022-01-01", end="2023-01-01")
        result['data'] = stock_data
    
    thread = threading.Thread(target=fetch_data)
    thread.start()
    thread.join(timeout=10)
    
    if thread.is_alive():
        return "Error: Fetching stock data took too long. Please try again.", None, None
    
    if result['data'] is None or result['data'].empty:
        return f"Error: Could not fetch data for {stock_ticker}. Please try again.", None, None

    stock_data = result['data'][['Adj Close']].copy()
    stock_data['Prediction'] = stock_data['Adj Close'].shift(-30)
    X = np.array(stock_data.drop(['Prediction'], axis=1))[:-30]
    y = np.array(stock_data['Prediction'])[:-30]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_prices = model.predict(X_test)
    
    # Create the ML output summary
    ml_output_summary = f"""
    The stock {stock_ticker} is showing a predicted trend based on historical data using machine learning. 
    The model anticipates the stock price will {'increase' if future_prices[-1] > X_test[-1][0] else 'decrease'} 
    over the next 30 days. Current indicators suggest a potential {'bullish' if future_prices[-1] > X_test[-1][0] else 'bearish'} movement.
    """

    try:
        # Generate a summary with Cohere API
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=ml_output_summary,
            max_tokens=150,
            temperature=0.7
        )
        cohere_analysis = response.generations[0].text.strip()
        print("Cohere Analysis:", cohere_analysis)  # Debugging output
    except Exception as e:
        return f"Error: Could not generate the summary due to an API issue. Details: {str(e)}", None, None

    return ml_output_summary, cohere_analysis  # Return both analyses


# Route for API key input page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['api_key'] = request.form['api_key']
        return redirect(url_for('analyze'))
    return render_template('api_key.html')

# Route to handle stock analysis
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'api_key' not in session:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        stock_ticker = request.form['ticker'].upper()
        co = cohere.Client(session['api_key'])  # Use the user-provided API key
        ml_analysis, cohere_analysis = fetch_and_analyze_stock(stock_ticker, co)  # Unpack both analyses
        return render_template('analyze.html', ml_result=ml_analysis, cohere_result=cohere_analysis)  # Pass both results

    return render_template('analyze.html')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
