import json

import pandas as pd
import yfinance as yf 


STOCKS_FILES = [
    ("train/configs/nyse_stocks.csv", "NYSE"),
    ("train/configs/nasdaq_stocks.csv", "NASDAQ"),
    ("train/configs/amex_stocks.csv", "AMEX")
]

if __name__ == "__main__":
    stocks = [
      ["U-UN", "TSX"],
      ["EFR", "TSX"],
      ["PDN", "ASX"],

      ["TER", "ASX"],
      ["OGC", "TSX"],
      ["CEY", "LSE"],
      ["PAAS", "TSX"],
      ["NST", "ASX"],
      ["AGI", "TSX"],
      ["EDV", "TSX"],
      ["BTG", "AMEX"],
      ["IAG", "NYSE"],
      ["EVN", "ASX"],
      ["LUG", "TSX"],
      ["HL", "NYSE"],
      ["OR", "TSX"],
      ["FIL", "TSX"],
      ["CDE", "NYSE"],
      ["EQX", "NYSE"],
      ["SAND", "NYSE"],
      ["DPM", "NYSE"],
      ["CG", "TSX"],
      ["KNT", "TSX"],
      ["ORLA", "NYSE"],
      ["NGD", "NYSE"],
      ["SILV", "NYSE"],
      ["GOR", "ASX"],
      ["RED", "ASX"],
      ["NG", "TSX"],
      ["NFGC", "NYSE"],
      ["SKE", "NYSE"],
      ["MUX", "NYSE"],
      ["IAUX", "NYSE"],
      ["GAU", "NYSE"],
      ["ARIS", "TSX"],
      ["AOT", "TSX"],

      ["STLC", "TSX"],
      ["SJT", "NYSE"],
      ["NRT", "NYSE"],
      ["PBT", "NYSE"],
      ["SBR", "NYSE"],
      ["PVL", "NYSE"],

      ["NHC", "ASX"]
    ]
    sectors = {}

    for ticker, exchange in stocks:
        if exchange == "ASX":
            yticker = f"{ticker}.AX"
        elif exchange == "TSX":
            yticker = f"{ticker}.TO"
        elif exchange == "LSE":
            yticker = f"{ticker}.L"

        if ticker in ("SJT", "NRT", "PBT", "SBR", "PVL"):
            sectors[f"{ticker}_{exchange}"] = "Energy"
        else:
            sectors[f"{ticker}_{exchange}"] = yf.Ticker(yticker).info["sector"]

    for filename, exchange in STOCKS_FILES:
        df = pd.read_csv(filename)
        df = df[df["Market Cap"] > 6e9]
        df = df[~df["Name"].str.contains("%")]
        
        for idx, row in df.iterrows():
            if row["Sector"] and not pd.isna(row["Sector"]):
                sectors[f'{row["Symbol"]}_{exchange}'] = row["Sector"]
            else:
                try:
                    sectors[f'{row["Symbol"]}_{exchange}'] = yf.Ticker(row["Symbol"]).info["sector"]
                except Exception as ex:
                    print(ex)
                    continue
            
            stocks.append([row["Symbol"], exchange])
    
    with open("train/configs/american_stocks.json", "w") as f:
        json.dump(
            {
                "tickers": stocks,
                "sectors": sectors
            },
            f
        )