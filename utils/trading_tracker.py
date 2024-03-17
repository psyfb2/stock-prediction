from typing import List

class TradingTracker:
    def __init__(self, capital: float, start_price: float, end_price: float, trades: List[float]):
        self.capital = capital
        self.start_price = start_price
        self.end_price = end_price
        self.trades = trades
    
    def calc_bh_profit(self):
        end_capital = (self.capital / self.start_price) * self.end_price
        return ((end_capital - self.capital) / self.capital) * 100
    
    def calc_trade_profit(self):
        end_capital = self.capital
        for buy, sell in self.trades:
            end_capital = (end_capital / buy) * sell
        return ((end_capital - self.capital) / self.capital) * 100
    
    def win_perc(self):
        return (sum(sell > buy for buy, sell in self.trades) / len(self.trades)) * 100
    
    def avg_profit(self):
        return sum(((sell - buy) / buy) * 100 for buy, sell in self.trades) / len(self.trades)
    
    def worst_trade(self):
        return min(((sell - buy) / buy) * 100 for buy, sell in self.trades)
    
    def best_trade(self):
        return max(((sell - buy) / buy) * 100 for buy, sell in self.trades)
    
    def bought_lower(self):
        return ((sum(self.trades[i + 1][0] < self.trades[i][1] for i in range(len(self.trades) - 1)) + (self.trades[0][0] < self.start_price)) / len(self.trades)) * 100


if __name__ == "__main__":
    tracker = TradingTracker(
        capital=10000, 
        start_price=14.89, end_price=14.47, 
        trades=[
            (13.64, 13.79),  # sold on resistance (100day SMA, ub, srsi_ob, 4 increasing closes)
            (13.51, 15.23),  # sold on resistancce (200day SMA, ub, srsi_ob, 5 increasing closes)
            (15.32, 13.93),  # selling on down trend (broke all SMA's, hugging lb)
            (13.93, 14.84),  # sold on resistance (200day SMA)
            (14.64, 13.11),  # selling on big down day (big volume)
            (14.84, 14.13),  # bbands squeeze to the downside
            (13.38, 14.29),  # selling on resistance (SMAs, ub, SRSI_OB_signal)
            (14.09, 12.84),  # selling on resistance (20daySMA)
            (12.33, 13.17),  # selling on resistance (50day SMA, ub, SRSI_OS)
            (12.28, 12.94),  # selling on resistance (50day SMA)
        ]
    )
    
    print(f"B&H Profit:           {round(tracker.calc_bh_profit(), 2)}%")
    print(f"Profit:               {round(tracker.calc_trade_profit(), 2)}%")

    print(f"Win Rate:             {round(tracker.win_perc(), 2)}%")
    print(f"Bought Lower Rate:    {round(tracker.bought_lower(), 2)}%")
    print(f"Avg Profit per Trade: {round(tracker.avg_profit(), 2)}%")
    print(f"Worst Trade:          {round(tracker.worst_trade(), 2)}%")
    print(f"Best Trade:           {round(tracker.best_trade(), 2)}%")
    print(f"Num Trades:           {len(tracker.trades)}")
    print("----------------------------")
