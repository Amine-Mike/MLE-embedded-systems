import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import joblib

from transpile_simple_model import transpile


def build_linear(X, y):
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "linear.joblib")


def build_logistic(X, y):
    threshold = y.mean()
    y_binary = (y > threshold).astype(int)
    model = LogisticRegression()
    model.fit(X, y_binary)
    joblib.dump(model, "logistic.joblib")


def build_tree(X, y):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    joblib.dump(model, "tree.joblib")


def build_models():
    df = pd.read_csv("houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"]
    build_linear(X, y)
    build_logistic(X, y)
    build_tree(X, y)


def main():
    build_models()
    transpile("linear.joblib", "model_code.c", test_input=[22.0, 1.0, 0.0])


if __name__ == "__main__":
    main()
