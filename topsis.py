
import sys
import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, result_file):
    try:
        # Read Data
        df = pd.read_csv(input_file)
        
        if len(df.columns) < 3:
            raise Exception("Input file must have at least 3 columns.")

        # Extract numeric part (assuming 1st col is name)
        data = df.iloc[:, 1:].values.astype(float)
        
        # Parse weights and impacts
        w = [float(i) for i in weights.split(',')]
        imp = impacts.split(',')

        if len(w) != data.shape[1] or len(imp) != data.shape[1]:
            raise Exception("Number of weights/impacts must match criteria count.")

        # 1. Vector Normalization
        rss = np.sqrt(np.sum(data**2, axis=0))
        norm_data = data / rss

        # 2. Weighted Normalization
        weighted_data = norm_data * w

        # 3. Ideal Best & Worst
        ideal_best = []
        ideal_worst = []
        
        for i in range(len(imp)):
            if imp[i] == '+':
                ideal_best.append(np.max(weighted_data[:, i]))
                ideal_worst.append(np.min(weighted_data[:, i]))
            elif imp[i] == '-':
                ideal_best.append(np.min(weighted_data[:, i]))
                ideal_worst.append(np.max(weighted_data[:, i]))
            else:
                raise Exception("Impacts must be '+' or '-'.")

        # 4. Distance Calculation
        dist_best = np.sqrt(np.sum((weighted_data - ideal_best)**2, axis=1))
        dist_worst = np.sqrt(np.sum((weighted_data - ideal_worst)**2, axis=1))

        # 5. Topsis Score
        score = dist_worst / (dist_best + dist_worst)

        # 6. Saving Results
        df['Topsis Score'] = score
        df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)
        
        df.to_csv(result_file, index=False)
        print(f"Result file saved as {result_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    else:
        topsis(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
