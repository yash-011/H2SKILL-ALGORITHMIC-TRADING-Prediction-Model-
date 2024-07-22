
import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No stock data found for {ticker} from {start_date} to {end_date}")
        else:
            print(f"Successfully fetched stock data for {ticker} from {start_date} to {end_date}")
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()
