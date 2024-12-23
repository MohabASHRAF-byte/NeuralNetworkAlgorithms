import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import random

def encodeDataAndSplit(filtered_df, categories: list, sample_size: int = 30, normalize: bool = False):
    # Generate numeric encoding for categories
    category_encoding = {category: idx for idx, category in enumerate(categories)}

    # Prepare training and testing datasets
    train_dfs = []
    random_state = random.randint(30, 100000) % random.randint(30, 200)

    for category in categories:
        # Filter and sample rows for the current category
        df_category = filtered_df[filtered_df['bird category'] == category].sample(
            sample_size, random_state=random_state
        )
        df_category['category_encoded'] = category_encoding[category]
        train_dfs.append(df_category)

    # Concatenate sampled rows to form the training dataset
    df_train = pd.concat(train_dfs)

    # Prepare the test dataset by excluding the training samples
    df_test = filtered_df[
        filtered_df['bird category'].isin(categories)
    ].drop(df_train.index)

    # Encode test dataset categories
    df_test['category_encoded'] = df_test['bird category'].map(category_encoding)

    # Drop 'bird category' from both DataFrames
    df_train = df_train.drop(columns=['bird category']).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.drop(columns=['bird category']).sample(frac=1, random_state=42).reset_index(drop=True)

    # Normalize data if requested
    if normalize:
        # Compute min and max only on the training set
        train_min = df_train.drop(columns=['category_encoded']).min()
        train_max = df_train.drop(columns=['category_encoded']).max()

        # Apply normalization to both training and test sets using training min and max
        for column in df_train.columns:
            if column != 'category_encoded':  # Do not normalize the target column
                if train_max[column] - train_min[column] > 0:  # Avoid division by zero
                    df_train[column] = (df_train[column] - train_min[column]) / (train_max[column] - train_min[column])
                    df_test[column] = (df_test[column] - train_min[column]) / (train_max[column] - train_min[column])

    # Convert DataFrames to numpy arrays
    train_output = df_train['category_encoded'].values
    train_input = df_train.drop(columns=["category_encoded"]).values
    test_output = df_test['category_encoded'].values
    test_input = df_test.drop(columns=["category_encoded"]).values

    # One-hot encode the outputs
    train_output2 = []
    for i in train_output:
        ret = [0 for _ in range(len(categories))]
        ret[i] = 1
        train_output2.append(ret)

    test_output2 = []
    for i in test_output:  # Fix to encode test_output instead of train_output again
        ret = [0 for _ in range(len(categories))]
        ret[i] = 1
        test_output2.append(ret)

    train_output = train_output2
    test_output = test_output2

    return train_input, train_output, test_input, test_output
class Data:
    def __init__(self):
        self.df = pd.read_csv("DataSets/birds.csv")
        self.age_encoder = LabelEncoder()
        self.birdsCategories = ["A", "B", "C"]
        self.clean()

    def clean(self):
        # Fill null values with the mode in bird category
        most_frequent_gender = self.df.groupby('bird category')['gender'].apply(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'unknown')
        )

        # Reset the index to align with the main DataFrame
        most_frequent_gender = most_frequent_gender.reset_index(drop=True)
        self.df['gender'] = most_frequent_gender
        self.encode_age()

    def encode_age(self):
        # Assuming there's an 'age' column to encode
        if 'gender' in self.df.columns:
            self.df['gender'] = self.age_encoder.fit_transform(self.df['gender'])

    def decode_age(self, encoded_values):
        # Decode the encoded age values
        return self.age_encoder.inverse_transform(encoded_values)

    def GenerateData(self, categories: list, filtered_df= None):
        if filtered_df is None:
            filtered_df = self.df
        # Validate input categories
        if not all(cat in self.birdsCategories for cat in categories) :
            raise ValueError("Invalid Categories")

        return encodeDataAndSplit(filtered_df, categories,normalize=True)

    def GenerateDataWithFeatures(self, categories: list, features: list):
        # Validate input features
        if not all(feature in self.df.columns for feature in features):
            raise ValueError("One or more specified features are not in the dataset.")

        # Select only the specified features and 'bird category'
        df_filtered = self.df[features + ['bird category']]
        return self.GenerateData(categories, filtered_df=df_filtered)