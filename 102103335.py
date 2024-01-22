import sys
import pandas as pd
import numpy as np

def check_params():
    if len(sys.argv) != 5:
        print("Write it in correct format: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

def load_data(input_file):
    try:
        data = pd.read_csv(input_file)
        return data
    except FileNotFoundError:
        print("Error: File not found. Please provide a valid input file.")
        sys.exit(1)

def check_columns(data):
    if len(data.columns) < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)

def check_numeric(data):
    non_numeric_cols = data.iloc[:, 1:].applymap(lambda x: not np.isreal(x)).any()
    if non_numeric_cols.any():
        print("Error: Columns from 2nd to last must contain numeric values only.")
        sys.exit(1)

def check_weights_impacts(weights, impacts, num_cols):
    weights_list = list(map(int, weights.split(',')))
    impacts_list = impacts.split(',')

    if len(weights_list) != num_cols - 1 or len(impacts_list) != num_cols - 1:
        print("Error: Number of weights, impacts, and columns must be the same.")
        sys.exit(1)

    if not all(impact in ['+', '-'] for impact in impacts_list):
        print("Error: Impacts must be either +ve or -ve.")
        sys.exit(1)

def normalize_data(data):
    norm_data = data.iloc[:, 1:].apply(lambda x: x / np.sqrt(np.sum(x**2)), axis=0)
    return norm_data

def calculate_topsis_score(data, weights, impacts):
    norm_data = normalize_data(data)
    weighted_norm_data = norm_data * list(map(int, weights.split(',')))
    ideal_best = weighted_norm_data.max() if impacts[0] == '+' else weighted_norm_data.min()
    ideal_worst = weighted_norm_data.min() if impacts[0] == '+' else weighted_norm_data.max()
    topsis_score = np.sqrt(np.sum((weighted_norm_data - ideal_best)**2, axis=1)) / (
            np.sqrt(np.sum((weighted_norm_data - ideal_best)**2, axis=1)) +
            np.sqrt(np.sum((weighted_norm_data - ideal_worst)**2, axis=1))
    )
    return topsis_score

def main():
    check_params()
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    result_file = sys.argv[4]

    data = load_data(input_file)
    check_columns(data)
    check_numeric(data)
    check_weights_impacts(weights, impacts, len(data.columns))

    topsis_score = calculate_topsis_score(data, weights, impacts)
    data['Topsis Score'] = topsis_score
    data['Rank'] = data['Topsis Score'].rank(ascending=False)

    data.to_csv(result_file, index=False)
    print("Results saved to", result_file)

if __name__ == "__main__":
    main()
