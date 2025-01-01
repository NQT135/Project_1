import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start_date, end_date):
    # Fetch data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset index to get 'Date' as a column
    stock_data.reset_index(inplace=True)
    
    # Rename columns to match the desired format
    stock_data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # Add a 'Name' column with the ticker symbol
    stock_data['Name'] = ticker
    
    return stock_data[['date', 'open', 'high', 'low', 'close', 'volume', 'Name']]

def main():
    # Define the stock ticker symbol, start date, and end date
    ticker = 'AMD' 
    start_date = '2020-01-01'
    end_date = '2024-12-31'

    # Fetch the stock data
    data = fetch_stock_data(ticker, start_date, end_date)

    # Display the data
    print(data)

    # Define the folder path where you want to save the CSV file
    folder_path = 'D:/2024/Projects/Nguyen Quoc Thai 20225456 _ Project1/Stock-Prices-ML-Dashboard/individual_stocks_5yr'  # Change this to your desired folder path
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the data to a CSV file in the specified folder
    data.to_csv(os.path.join(folder_path, f'{ticker}_stock_data.csv'), index=False)

if __name__ == "__main__":
    main()