import os
import argparse
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile
from typing import Tuple

class LinearRegression:
    def __init__(self, path: str = "data.csv", lr: float = 0.01, epochs: int = 10000, min_delta: float = 1e-7):
        assert os.path.isfile(path), "[!] Input file does not exist, or is not a file."
        # Set parameters for training
        self.lr = lr
        self.epochs = epochs
        self.min_delta = min_delta

        # Initialize thetas
        self.theta_0, self.theta_1 = 0, 0
        self.tmp_theta_0, self.tmp_theta_1 = 0, 0

        # Read data from file
        self.data = pd.read_csv(path)
        self.org_x, self.org_y = self.data.iloc[:, 0].tolist(), self.data.iloc[:, 1].tolist()

        # Normalize the data to be between 0 and 1 (training stability / efficiency)
        self.max_x, self.max_y = max(self.org_x), max(self.org_y)
        self.min_x, self.min_y = min(self.org_x), min(self.org_y)
        self.x = [x / self.max_x for x in self.org_x]
        self.y = [y / self.max_y for y in self.org_y]

        # Session file for dynamic plotting
        self.session_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)

    def predict(self, n: float) -> float:
        """
        Predicts the value of y for the given x.
            -> y = theta_0 + (theta_1 * x)
        """
        return self.theta_0 + self.theta_1 * n

    def compute_gradients(self) -> Tuple[float, float]:
        """
        Computes the gradients for the current thetas.
            - tmp_t0 = learningRate * 1/m * sum(x - y)
            - tmp_t1 = learningRate * 1/m * sum((x - y) * x)
        """
        sum_t0, sum_t1 = 0, 0
        for x, y in zip(self.x, self.y):
            sum_t0 += (self.predict(x) - y)
            sum_t1 += (self.predict(x) - y) * x
        return (self.lr * (1 / len(self.x)) * sum_t0, self.lr * (1 / len(self.x)) * sum_t1)

    def fit(self) -> "LinearRegression":
        """
        Fits for n iterations (epochs) on the given data.
        """
        old_mse = self.mse()
        for i in range(self.epochs):
            # Compute temporary thetas
            t0, t1 = self.compute_gradients()

            # Update thetas
            self.theta_0 -= t0
            self.theta_1 -= t1

            # Log progress
            print(
                "\033[K\r" + "Fitting {:.2f}% / MSE: {:.3f} / R2: {:.3f}% (delta: {:.4f})".format(
                    ((i + 1) / self.epochs) * 100, self.mse(), self.r2() * 100, abs(self.mse() - old_mse)
                ),
                end="\r"
            )

            # Will be used to animate the training later
            if i % 5 == 0:
                # normalize thetas to be able to plot them
                tmp_t0 = self.theta_0 * self.max_y
                tmp_t1 = self.theta_1 * self.max_y / self.max_x
                self.session_file.write("{};{}\n".format(tmp_t0, tmp_t1))

            # Early stopping if the MSE is not improving
            if i > 0 and abs(self.mse() - old_mse) < self.min_delta:
                print("\n033[K\rEarly stopping at epoch {}.\n".format(i))
                break
            else:
                old_mse = self.mse()


        print("\nTraining summary:")
        print("  - MSE: {:.3f}".format(self.mse()))
        print("  - R2: {:.3f}".format(self.r2()))

        # Denormalize thetas to be able to predict values without re-normalizing them
        self.theta_0 *= self.max_y
        self.theta_1 *= self.max_y / self.max_x

        print("  - Theta 0: {:.3f}".format(self.theta_0))
        print("  - Theta 1: {:.3f}".format(self.theta_1))

        self.session_file.write("{};{}\n".format(self.theta_0, self.theta_1))
        self.session_file.close()
        return self

    def plot(self) -> "LinearRegression":
        """
        Plot X and Y values in a scatter plot.
        In the same plot, add points for each predictions of X
        """
        predictions = [self.predict(x) for x in self.org_x]
        sns.scatterplot(x=self.org_x, y=self.org_y, color="green")
        sns.lineplot(x=self.org_x, y=predictions, color="red")
        plt.legend(labels=["Data", "Predictions"])
        plt.title("Linear regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        return self

    def replay(self) -> "LinearRegression":
        print("[#] Replaying training, this should take few seconds...")
        all_predictions = []
        with open(self.session_file.name, "r") as f:
            for line in f:
                theta_0, theta_1 = line.strip().split(";")
                self.theta_0, self.theta_1 = float(theta_0), float(theta_1)
                all_predictions.append([self.predict(x) for x in self.org_x])
        # create an animated plot in the same way as plot
        fig = plt.figure()
        ax = plt.axes(xlim=(0, max(self.org_x)), ylim=(0, max(self.org_y)))
        line, = ax.plot([], [], lw=2)
        plt.legend(labels=["Data", "Predictions"])
        plt.title("Linear regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        def init():
            ax.scatter(self.org_x, self.org_y, color="green")
            line.set_data([], [])
            return line,
        def animate(i):
            line.set_data(self.org_x, all_predictions[i])
            ax.set_title("Linear regression (epoch {})".format(i * 5))
            return line,
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(all_predictions), interval=20, blit=True)
        print("[#] Rendering animation, this may take a while...")
        anim.save("animation.mp4", writer="ffmpeg")
        return self

    def mse(self) -> float:
        """
        Computes the mean squared error for the current thetas.
        """
        n = 0
        for x, y in zip(self.x, self.y):
            n += (self.predict(x) - y) ** 2
        return n / len(self.x)
    
    def r2(self) -> float:
        """
        Computes the R2 score for the current thetas, serves as a metric for the model's accuracy.
        """
        y_mean = sum(self.y) / len(self.y)

        # Total sum of squares
        ss_tot = sum([(y - y_mean) ** 2 for y in self.y])

        # Residual sum of squares
        ss_res = sum([(y - self.predict(x)) ** 2 for x, y in zip(self.x, self.y)])
        return 1 - (ss_res / ss_tot)

    def save(self, path: str = "thetas.pkl") -> "LinearRegression":
        """
        Save current thetas to a file (default: thetas.pkl)
        """
        with open(path, "wb+") as f:
            pickle.dump((self.theta_0, self.theta_1), f)
        return self

    def load(self, path: str = "thetas.pkl") -> "LinearRegression":
        """
        Loads current thetas to a file (default: thetas.pkl)
        Throws an error if the file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError("[!] Thetas file does not exist, or is not a file.")
        try:
            with open(path, "rb") as f:
                self.theta_0, self.theta_1 = pickle.load(f)
        except Exception as e:
            raise Exception("[!] Error while loading thetas: {}".format(e))
        return self

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", help="Input dataset file (.csv)", default="data.csv")
    args.add_argument("-l", "--lr", help="Learning rate", default=0.01, type=float)
    args.add_argument("-e", "--epochs", default=10000, help="Number of maximum iterations for fitting", type=int)
    args.add_argument("-p", "--plot", help="Plot the data", action="store_true", default=False)
    args.add_argument("-r", "--replay", help="Replay the training", action="store_true", default=False)
    args.add_argument("-s", "--simulate", help="Runs the training without saving thetas", action="store_true", default=False)
    args.add_argument("-m", "--min_delta", help="Minimum MSE delta for early stopping", default=1e-7, type=float)
    args.add_argument("-v", "--visualize", help="Visualize the data", action="store_true", default=False)
    args = args.parse_args()
    lr = LinearRegression(
        args.input,
        args.lr,
        args.epochs,
        args.min_delta,
    ).fit()
    if args.plot:
        lr.plot()
    if not args.simulate:
        lr.save()
    if args.replay:
        lr.replay()
