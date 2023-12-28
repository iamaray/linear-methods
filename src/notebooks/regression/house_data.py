import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class HouseData:
    def __init__(self):
        self.df = pd.read_csv("data/house_prices.csv")
        self.n = len(self.df.index)

    def setIndex(self, idxCol):
        self.n = self.df[idxCol].count()
        self.df.set_index(idxCol)

    def col(self, colName, numpyArr=False, bounds=None):
        if bounds != None:
            l, u = bounds
            return self.df[colName][l: u].to_numpy() if numpyArr == True else self.df[colName][l: u]
        else:
            return self.df[colName].to_numpy() if numpyArr == True else self.df[colName]

    def randomSample(self, frac):
        return self.df.sample(frac=frac)

    def split(self, testFrac, shuffle=True):
        train, test = train_test_split(
            self.df, test_size=testFrac, shuffle=shuffle)
        return (train, test)

    def plot(self, xCol, yCol, dataRange=None):
        x, y = self.col(xCol, True, dataRange), self.col(yCol, True, dataRange)
        plt.scatter(x, y)
        plt.xlabel(xCol)
        plt.ylabel(yCol)
        plt.show()
        plt.savefig(f'house_fig_{xCol}x{yCol}')
