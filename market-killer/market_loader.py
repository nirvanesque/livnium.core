import pandas as pd
import glob
import os

def load_market(path):
    files = glob.glob(os.path.join(path, "*.csv"))
    market = {}

    for f in files:
        symbol = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)

        # normalize columns
        df = df.rename(columns={
            'Close/Last': 'Close',
            'Volume': 'Volume'
        })

        # parse dates & sort
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
        
        market[symbol] = df
    
    return market




# usage:
# rom market_loader import load_market
# market = load_market("/Users/chetanpatil/Desktop/clean-nova-livnium/market-killer/market")
