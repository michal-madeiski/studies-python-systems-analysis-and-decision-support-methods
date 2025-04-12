import pandas as pd

def load_data(path):
    try:
        d = pd.read_csv(path, low_memory=False)
        return d
    except FileNotFoundError as e:
        return e

loaded_fifa22_data = load_data("../../data/fifa22_database.csv")


if __name__ == "__main__":
    print(loaded_fifa22_data)