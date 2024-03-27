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
        start_price=399.93, end_price=108.10,
        trades=[
            (399.93, 305.64), # selling on resistance (20sma, MFI divergence)
            (292.12, 254.68), # avoid downtrend (broke 200sma with strong volume, hugging lb)
            (316.64, 292.14), # big down day 
            (312.16, 262.37), # avoiding downtrend (hugging lb)
            (236.32, 209.39),
            (190.95, 209.64)
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
