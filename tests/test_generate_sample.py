import math
import pandas as pd
import numpy as np


def extract_mean_se(txt):
    mu = float(txt.split()[0])
    txt = txt.split()[1]
    txt = txt.split('(')[1]
    txt = txt.split(')')[0]
    sigma = float(txt)
    return mu, sigma


def main():
    df = pd.read_csv("/home/minhvu/github/vqa_idrid/data/TMI_2019_all.csv")
    df_data = df.iloc[:, 1:]
    data = df_data.values

    X = np.zeros((6*21, 9))

    for i in range(6):
        for j in range(9):
            mu, sigma = extract_mean_se(data[i][j])
            s = np.random.normal(mu, sigma, 21)
            for k in range(21):
                row = i*21+k
                X[row][j] = s[k]

    columns = ["MUTAN + tanh",
               "MLB + tanh",
               "MUTAN + tanh + Att",
               "MLB + tanh + Att",
               "G-MLB + relu + Att",
               "Ours + tanh",
               "Ours + tanh + Att",
               "Ours + relu",
               "Ours + relu + Att"
               ]

    print(X)

    df = pd.DataFrame(X, columns=columns)

    df.to_csv("/home/minhvu/github/vqa_idrid/data/TMI_2019_full.csv", index=False)


if __name__ == "__main__":
    main()
