import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(currency_name, split=0):
    """
    load data from csv file
    :param currency_name: currently available AAPL_train, AAPL_test, SPY_train, SPY_test
    :param split: if split != 0 the data is split between training and validation set
    :return: pandas dataframe of the training set/ 2 pandas dataframe in the case split != 0
    """
    data = pd.read_csv(f"data/{currency_name}.csv")
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    if split:
        train, val = train_test_split(data, test_size=split, shuffle=False)
        val.reset_index(inplace=True, drop=True)
        return train, val
    return data

