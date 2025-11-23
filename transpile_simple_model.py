from pathlib import Path
import joblib
import os
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def get_linear_code(model):
    coefs = model.coef_
    intercept = model.intercept_
    coef_str = ", ".join([f"{c}f" for c in coefs])

    return f"""
    float intercept = {intercept}f;
    float coefficients[] = {{{coef_str}}};
    float res = intercept;
    for (int i = 0; i < n_feature; i++)
        res += coefficients[i] * features[i];
    return res;
    """


def get_logistic_prelude():
    return """
    float exp_approx(float x, int n_term)
    {{
        float res = 0;
        float power = 1;
        int fact = 1;
        for (int i = 0; i <= n_term; ++i)
        {{
            res += power / fact;
            power *= x;
            fact *= i + 1;
        }}
        return res;
    }}

    float sigmoid(float x)
    {{
        return 1 / (1 + exp_approx(-x, 10));
    }}
    """


def get_logistic_code(model):
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    coef_str = ", ".join([f"{c}f" for c in coefs])

    return f"""
    float intercept = {intercept}f;
    float coefficients[] = {{{coef_str}}};
    float z = intercept;

    for (int i = 0; i < n_feature; i++)
        z += coefficients[i] * features[i];

    return sigmoid(z);
    """


def get_tree_code(model):
    """Generates C code for Decision Trees (Array traversal)."""
    tree = model.tree_

    left_children = tree.children_left
    right_children = tree.children_right
    thresholds = tree.threshold
    features = tree.feature

    values = [v[0][0] for v in tree.value]

    left_str = ", ".join(map(str, left_children))
    right_str = ", ".join(map(str, right_children))
    thresh_str = ", ".join([f"{t}f" for t in thresholds])
    feat_str = ", ".join(map(str, features))
    val_str = ", ".join([f"{v}f" for v in values])

    return f"""
    // Decision Tree Structure
    int children_left[] = {{{left_str}}};
    int children_right[] = {{{right_str}}};
    float thresholds[] = {{{thresh_str}}};
    int split_features[] = {{{feat_str}}};
    float node_values[] = {{{val_str}}};

    int node = 0;

    while (children_left[node] != -1) {{
        float val_to_check = features[split_features[node]];

        if (val_to_check <= thresholds[node]) {{
            node = children_left[node];
        }} else {{
            node = children_right[node];
        }}
    }}

    return node_values[node];
    """


def transpile(model_path, output_c_file, test_input):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = joblib.load(model_path)
    model_type = type(model).__name__
    print(f"Detected Model Type: {model_type}")

    prediction_logic = ""
    includes = "#include <stdio.h>\n"
    prelude = ""

    if isinstance(model, LinearRegression):
        prediction_logic = get_linear_code(model)
        n_features = len(model.coef_)
    elif isinstance(model, LogisticRegression):
        prediction_logic = get_logistic_code(model)
        n_features = len(model.coef_[0])
        prelude = get_logistic_prelude()
    elif isinstance(model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        prediction_logic = get_tree_code(model)
        n_features = model.n_features_in_
    else:
        print("Unsupported model type.")
        return

    input_str = ", ".join([f"{x}f" for x in test_input])

    c_content = f"""{includes}

{prelude}

// Prediction Function
float prediction(float *features, int n_feature) {{
    {prediction_logic}
}}

int main() {{
    float input_features[] = {{{input_str}}};
    int n_features = {n_features};

    float result = prediction(input_features, n_features);

    printf("Prediction: %.6f\\n", result);
    return 0;
}}
"""

    print(
        f"Python Code Prediction: {model.predict(np.array(test_input).reshape(1, -1))}"
    )

    with open(output_c_file, "w") as f:
        f.write(c_content)

    print(f"Generated {output_c_file}")
    exec_name = Path(output_c_file).stem

    cmd = f"gcc {output_c_file} -o {exec_name}"
    print(f"Compiling: {cmd}")
    if os.system(cmd) == 0:
        print("Running C executable:")
        os.system(f"./{exec_name}")


if __name__ == "__main__":
    # Linear Regression
    transpile("linear.joblib", "model_code.c", test_input=[22.0, 1.0, 0.0])

    # Logistic Regression
    transpile("logistic.joblib", "model_code.c", test_input=[22.0, 1.0, 0.0])

    # Decision Tree
    transpile("tree.joblib", "model_code.c", test_input=[22.0, 1.0, 0.0])
