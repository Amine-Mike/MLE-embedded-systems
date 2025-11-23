import joblib
import os
import numpy as np


def generate_c_code(model_path, output_c_file):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = joblib.load(model_path)

    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        print("Error: The loaded object is not a standard LinearRegression model.")
        return

    coefs = model.coef_
    intercept = model.intercept_
    n_features = len(coefs)

    input_features = [22, 1, 0]

    input_str = str(input_features)[1:-1]
    coef_str = ", ".join([f"{c}f" for c in coefs])

    c_content = f"""
#include <stdio.h>

float prediction(float *features, int n_feature) {{
    float intercept = {intercept}f;
    float coefficients[] = {{{coef_str}}};
    float res = intercept;

    for (int i = 0; i < n_feature; i++)
        res += coefficients[i] * features[i];
    return res;
}}

int main() {{
    float input_features[] = {{{input_str}}};
    int n_features = {n_features};

    float result = prediction(input_features, n_features);

    printf("C Code Prediction: %.6f\\n", result);
    return 0;
}}
"""
    print(
        f"Python Code Prediction: {model.predict(np.array(input_features).reshape(1, -1))}"
    )

    with open(output_c_file, "w") as f:
        f.write(c_content)

    compile_cmd = f"gcc {output_c_file} -o model_code"
    print(f"To compile manually, run:\n{compile_cmd}")

    exit_code = os.system(compile_cmd)

    if exit_code == 0:
        os.system("./model_code")


if __name__ == "__main__":
    generate_c_code("regression.joblib", "model_code.c")
