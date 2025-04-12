import pandas as pd
from data_loader import loaded_fifa22_data as data

def numerical_stats(d):
    num_data = d.select_dtypes(include=['number'])
    num_stats = pd.DataFrame()

    num_stats['mean'] = num_data.mean()
    num_stats['median'] = num_data.median()
    num_stats['min'] = num_data.min()
    num_stats['max'] = num_data.max()
    num_stats['std'] = num_data.std()
    num_stats['5th percentile'] = num_data.quantile(0.05)
    num_stats['95th percentile'] = num_data.quantile(0.95)
    num_stats['missing values'] = num_data.isnull().sum()

    num_stats.index.name = "column_name"

    return num_stats

def categorical_stats(d):
    cat_data = d.select_dtypes(include=['object', 'category'])
    cat_stats = pd.DataFrame()

    cat_stats['unique classes'] = cat_data.nunique()
    cat_stats['missing values'] = cat_data.isnull().sum()
    cat_stats['class proportions'] = cat_data.apply(lambda x: x.value_counts(normalize=True).to_dict())

    cat_stats.index.name = "column_name"

    return cat_stats

def save_numerical_stats(d):
    numerical_stats(d).to_csv("../data/num_stats.csv", index=True)

def save_categorical_stats(d):
    categorical_stats(d).to_csv("../data/cat_stats.csv", index=True)


if __name__ == "__main__":
    save_numerical_stats(data)
    save_categorical_stats(data)