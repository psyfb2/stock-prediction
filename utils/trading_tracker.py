from typing import List, Tuple

class TradingTracker:
    def __init__(self, capital: float, start_price: float, end_price: float, trades: List[Tuple[float, float, int]], total_candles: int):
        self.capital = capital
        self.start_price = start_price
        self.end_price = end_price
        self.trades = trades
        self.total_candles = total_candles
    
    def calc_bh_profit(self):
        end_capital = (self.capital / self.start_price) * self.end_price
        return ((end_capital - self.capital) / self.capital) * 100
    
    def calc_trade_profit(self):
        end_capital = self.capital
        for buy, sell, _ in self.trades:
            end_capital = (end_capital / buy) * sell
        return ((end_capital - self.capital) / self.capital) * 100

    def calc_profit_factor(self):
        end_capital = self.capital
        gains = losses = 0
        for buy, sell, _ in self.trades:
            new_end_capital = (end_capital / buy) * sell
            change = new_end_capital - end_capital
            end_capital = new_end_capital

            if change < 0:
                losses += -change
            elif change > 0:
                gains += change

        return gains / losses if losses > 0 else float("inf")
    
    def trade_profits(self):
        return [((sell - buy) / buy) * 100 for buy, sell, _ in self.trades]
    
    def win_perc(self):
        return (sum(sell >= buy for buy, sell, _ in self.trades) / len(self.trades)) * 100
    
    def avg_profit(self):
        return sum(((sell - buy) / buy) * 100 for buy, sell, _ in self.trades) / len(self.trades)
    
    def worst_trade(self):
        return min(((sell - buy) / buy) * 100 for buy, sell, _ in self.trades)
    
    def best_trade(self):
        return max(((sell - buy) / buy) * 100 for buy, sell, _ in self.trades)
    
    def bought_lower(self):
        return ((sum(self.trades[i + 1][0] < self.trades[i][1] for i in range(len(self.trades) - 1)) + (self.trades[0][0] < self.start_price)) / len(self.trades)) * 100
    
    def avg_trade_length(self):
        return sum(trade[2] for trade in self.trades) / len(self.trades)
    
    def exposure(self):
        return sum(trade[2] for trade in self.trades) / self.total_candles


if __name__ == "__main__":
    tracker = TradingTracker(
        capital=10000, 
        start_price=336.92, end_price=580.07, total_candles=252,
        trades=[
            (515.83, 552.16, 24),  # uptrend SRSI OS
        ]
    )
    profit = tracker.calc_trade_profit()
    exposure = tracker.exposure()
    
    print(f"B&H Profit:           {round(tracker.calc_bh_profit(), 2)}%")
    print(f"Risk Adjusted Return: {round(profit / exposure, 2)}%")
    print(f"Profit:               {round(tracker.calc_trade_profit(), 2)}%")
    print(f"Exposure:             {round(exposure * 100, 2)}%")
    print(f"Profit Factor:        {round(tracker.calc_profit_factor(), 2)}")

    print(f"Win Rate:             {round(tracker.win_perc(), 2)}%")
    print(f"Avg Profit per Trade: {round(tracker.avg_profit(), 2)}%")
    print(f"Avg Length of Trade:  {round(tracker.avg_trade_length(), 2)}")
    print(f"Best Trade:           {round(tracker.best_trade(), 2)}%")
    print(f"Worst Trade:          {round(tracker.worst_trade(), 2)}%")
    print(f"Num Trades:           {len(tracker.trades)}")
    print(f"Bought Lower Rate:    {round(tracker.bought_lower(), 2)}%")
    print(f"Trade Profits:\n{[round(p, 2) for p in tracker.trade_profits()]}")
    print("----------------------------")

    # The average yearly profit for the S&P 500 is around 8% (not including dividends)

    # import yfinance as yf
    # import numpy as np
    # import matplotlib.pyplot as plt
    # spx = yf.Ticker("^SPX")
    # df = spx.history(period="max")
    # print(df)
    # # calculate avg daily return
    # c = df['Close']
    # pct_change = 100 * c.pct_change()[1:] 
    # expected_daily_return = pct_change.mean()
    # std_daily_return      = pct_change.std()
    # print("E[X], std(X):", expected_daily_return, std_daily_return)

    # q75, q25 = np.percentile(pct_change.values, [75 ,25])
    # iqr = q75 - q25
    # bw = 2 * iqr / len(pct_change) ** (1 / 3)
    # min_, max_ = pct_change.min(), pct_change.max()
    # print("min, max:", min_, max_)
    # print("bw:", bw)

    # plt.hist(pct_change.values, bins=np.arange(min_, max_ + bw, bw))
    # plt.xlabel("Daily return (%)")
    # plt.ylabel("Frequency")
    # plt.title("SPX Daily Returns (%) Histogram")
    # plt.show()
    # expected_daily_return = 0.030579194097271247
    # print("Average Annual Return (%):", (((1 + expected_daily_return /  100) ** 252) - 1) * 100)