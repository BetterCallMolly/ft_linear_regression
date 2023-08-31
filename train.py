import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
from time import sleep

"""
TODOs for bonuses :
    - [ ] Automatic learning rate (alpha) calculation
    - [ ] Dynamic plotting (update a plot for each iteration of the training?)
    - [ ] Early stopping (goal?)
    - [ ] MSE calculation (for early stopping?)
    - [ ] Multi feature support
"""

"""
TODOs :
    - [ ] Implement training
    - [ ] Implement plotting
    - [x] Save thetas to pickle
    - [x] Load thetas from pickle
"""

"""
Formulas used :
    - y = theta_0 + (theta_1 * x)
    - tmp_theta_0 = learningRate * 1/m * sum(x - y)
    - tmp_theta_1 = learningRate * 1/m * sum((x - y) * x)
    (where m is the number of samples in the data)
"""

class LinearRegression:
    def __init__(self, path: str, lr: float, epochs: int):
        assert os.path.isfile(path), "[!] Input file does not exist, or is not a file."
        self.path = path
        self.lr = lr
        self.epochs = epochs
        self.data = pd.read_csv(self.path)
        self.x, self.y = self.data.iloc[:, 0].tolist(), self.data.iloc[:, 1].tolist()
        self.theta_0, self.theta_1 = 0, 0
        self.tmp_theta_0, self.tmp_theta_1 = 0, 0
        self.min_x = min(self.x)
        self.max_x = max(self.x)
        self.min_y = min(self.y)
        self.max_y = max(self.y)
        print(f"[+] Loaded data at {path}, m={len(self.x)}")

    def predict(self, n: float) -> float:
        """
        Predicts the value of y for the given x.
        """
        return self.theta_0 + self.theta_1 * n

    def compute_gradients(self) -> Tuple[float, float]:
        """
        Computes the gradients for the current thetas.
        """
        sum_t0, sum_t1 = 0, 0
        for x, y in zip(self.x, self.y):
            sum_t0 += (self.predict(x) - y)
            sum_t1 += (self.predict(x) - y) * x
        self.tmp_theta_0 = self.lr * (1 / len(self.x)) * sum_t0
        self.tmp_theta_1 = self.lr * (1 / len(self.x)) * sum_t1
        return (self.tmp_theta_0, self.tmp_theta_1)

    def fit(self) -> "LinearRegression":
        """
        Fits for n iterations (epochs) on the given data.
        """
        for i in range(self.epochs):
            self.theta_0, self.theta_1 = self.tmp_theta_0, self.tmp_theta_1
            t0, t1 = self.compute_gradients()
            self.tmp_theta_0 -= t0
            self.tmp_theta_1 -= t1
            # TODO: compute new theta
        print()
        print(f"[+] Trained for {self.epochs} epochs, theta_0={self.theta_0}, theta_1={self.theta_1}")
        return self

    def save(self, path: str = "thetas.pkl") -> "LinearRegression":
        """
        Save current thetas to a file (default: thetas.pkl)
        """
        with open(path, "wb+") as f:
            pickle.dump((self.theta_0, self.theta_1), f)
        return self

    def load(self, path: str = "thetas.pkl") -> "LinearRegression":
        """
        Loads current thetas to a file (default: thetas.pkl), and returns them
        Has no effects if the file doesn't exists
        """
        if not os.path.isfile(path):
            return self
        with open(path, "rb") as f:
            self.theta_0, self.theta_1 = pickle.load(f)
        return self

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", help="Input dataset file (.csv)", default="data.csv")
    args.add_argument("-l", "--lr", help="Learning rate", default=0.0001, type=float)
    args.add_argument("-e", "--epochs", default=100000, help="Number of iterations for fitting", type=int)
    args = args.parse_args()
    linear_regression = LinearRegression(args.input, args.lr, args.epochs)
    linear_regression.fit()
    print(linear_regression.predict(144500))