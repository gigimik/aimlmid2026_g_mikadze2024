import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pearson_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)
    )

    return numerator / denominator

def main():
    df = pd.read_csv("points.csv")

    x = df["x"].values
    y = df["y"].values

    r = pearson_correlation(x, y)
    print(f"Pearson correlation coefficient r = {r:.6f}")

    # Visualization
    plt.figure()
    plt.scatter(x, y, color="blue", label="Observed data points")

    # regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color="red", label="Best-fit line")

    plt.title(f"Pearson Correlation (r = {r:.3f})")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("correlation_plot.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
