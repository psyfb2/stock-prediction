# Stock Market Prediction

Source code for the stock market prediction tool hosted at []. The aim was to create a tool which gives a buy rating between 0-100 for any given stock. To this end, Machine Learning was used to make the predictions. Concretely, the problem was posed as a binary classification problem where each candle on the daily chart of over 500 stocks (mainly S&P 500 stocks) was classified according to:

- label 1: if a Take-Profit order of 6% is hit before a Trailing-Stop-Loss order of 6% relative to the next candles open price.
- label 0: If Trailing-Stop-Loss order is hit first.

The features corresponding to a candles label were a number of Technical Indicators for the previous T candles inclusive of the current candle. Only Technical indicators which are roughly stationary were used, this is so that the features were comparable and retained the same meaning regardless of which stock they came from. Concretely, the problem is framed as a time series classification problem where each input has shape (T, D) and has a binary label.

The training dataset had approximately 1.4 million samples from 500+ stocks using data from 2012-01-01 to 2023-04-01, with around 55% of the samples having label 0. The validation set had around 50,000 samples using data from 2023-04-01 to 2023-09-01, with around 57% of the samples having label 0. The test set had around 40,000 samples using data from 2023-09-01 to 2024-01-06, with around 56% of the samples having label 0.

Two models were considered:

- Multi-layer Perceptron (MLP).
- Classification Transformer. This is an encoder only Transformer with a classification head.

For each model hyperparameters were tuned on the validation set and the best threshold was found using the validation ROC curve. Thereafter, the best model from the validation stage was tested using the test set. An MLP with hidden layers [64, 32, 12] using T=2 got the lowest validation loss and the test set results are given below.

## Test Set Results

Using best threshold = 0.40, which calculated using validation ROC:

|                     | Precision | Recall |  F1  | Support |
| :-----------------: | :-------: | :----: | :--: | :-----: |
| Don't Buy (label 0) |   0.72    |  0.51  | 0.60 |  22170  |
|    Buy (label 1)    |   0.55    |  0.75  | 0.63 |  17509  |
|      Accuracy       |           |        | 0.62 |  39679  |
|      Macro Avg      |   0.63    |  0.63  | 0.61 |  39679  |
|    Weighted Avg     |   0.64    |  0.62  | 0.61 |  39679  |

Test Set ROC Curve:
![roc_curve](https://github.com/psyfb2/stock-prediction/blob/main/test_roc.png?raw=true)

# Lessons Learnt

According to the Efficient Market Hypothesis and Random Walk Hypothesis, past stock prices have no effect on future stock prices, essentially technical analysis is useless. Others would have you believe that stock prices follow predictable patterns due to them being a function of human phycology.

The truth seems to be somewhere in the middle. The trained ML model was able to achieve 62% accuracy on the test set. However, if stock prices were truly random then one would expect the accuracy to be around 55% (55% of the test-set samples had label 0). Also as can be seen by the test-set ROC, the ML model outperforms random predictions (shown by the dotted blue line). Of-course the difference is not much, but this seems to suggest for the most part stock prices are random but their is some signal and predictability there. My hunch stock prices become less random during key events on the chart (i.e. hitting support or resistance) and this is where the model is able to outperform random predictions. Furthermore, stock prices usually have a trend, so the prices could be random but have a bias towards the past trend.

Therefore, in my humble opinion, the best approach to the markets is identifying companies which are undervalued using fundamental analysis, Buffet style, and then choosing a good entry point by using key events on a chart level and the underlying trend.

The models could be further improved by:

- Having a separate model for each category of stocks (e.g. tech, gold, oil & gas, etc). The way in which these stocks move could be different on some fundamental level so perhaps having a separate model for each could help.
- Using reinforcement learning. Instead of providing a label, which is in some sense arbitrary, since the Take-Profit and Trailing-Stop-Loss levels of 6% could have been any other percentage, the model should learn to maximize change in portfolio value.

## Running Locally

To train a model locally run:

```
python -m train.train_supervised  -c <path_to_train_config>
```

You can customise the default train config in train/configs to change the stocks to use, features to use, model hyperparameters, etc. A directory will be created within training_artifacts for this training which stores logs, saved model, etc.

To run the api locally run the following within the api directory:

```
uvicorn main:app --reload
```

The environment variable 'MODEL_FILES_DIR' must be set. This should point to the directory which was created during training. Also the environment variable 'MONGO_DB_URI' should be set to your Mongo DB URI. The Mongo DB should have a database with the name 'sm_api_db' and collection 'probability_cache'. The collection 'probability_cache' needs to have a TTL index named 'expiresAt' with expiresAfterSeconds=0.

## License

[MIT](https://choosealicense.com/licenses/mit/)
