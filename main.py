import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

from transpile_simple_model import generate_c_code


def build_model():
    df = pd.read_csv("houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "regression.joblib")


def main():
    build_model()
    generate_c_code("regression.joblib", "model_code.c")


if __name__ == "__main__":
    main()
