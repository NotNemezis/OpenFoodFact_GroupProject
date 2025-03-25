import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
import hashlib

class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the DataPreprocessor with the dataset path.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")

    def drop_columns_and_handle_missing(self, columns_to_drop=None, missing_value_threshold=0.5):
        """
        Drop specified columns and handle missing values by dropping columns with missing values above the threshold.
        """
        if columns_to_drop:
            print(f"Dropping columns: {columns_to_drop}")
            self.data = self.data.drop(columns=columns_to_drop, errors='ignore')

        if missing_value_threshold is not None:
            print(f"Dropping columns with more than {missing_value_threshold * 100}% missing values...")
            missing_ratio = self.data.isnull().mean()
            columns_to_remove = missing_ratio[missing_ratio > missing_value_threshold].index
            print(f"Columns to remove due to missing values: {list(columns_to_remove)}")
            self.data = self.data.drop(columns=columns_to_remove)

    def impute_missing_values(self, imputation_strategy='mean', knn_neighbors=5):
        """
        Handle missing values using the specified imputation strategy, based on the column types.
        """
        print(f"Filling missing values for numerical data using the '{imputation_strategy}' strategy...")
        
        # Impute numerical columns
        if imputation_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=knn_neighbors)
        else:
            imputer = SimpleImputer(strategy=imputation_strategy)
        self.data[self.numerical_columns] = imputer.fit_transform(self.data[self.numerical_columns])
        
        # Impute non-numerical columns
        print("Filling missing values for non-numerical data using the 'most_frequent' strategy...")
        imputer = SimpleImputer(strategy='most_frequent')
        self.data[self.non_numerical_columns] = imputer.fit_transform(self.data[self.non_numerical_columns])

    def determine_column_types(self):
        """
        Determine numerical and non-numerical columns in the dataset.
        """
        self.numerical_columns = self.data.select_dtypes(include=['number']).columns.tolist()
        self.non_numerical_columns = self.data.select_dtypes(exclude=['number']).columns.tolist()
        print(f"Numerical columns: {self.numerical_columns}")
        print(f"Non-numerical columns: {self.non_numerical_columns}")

    def encode_non_numerical_columns(self):
        """
        Encode non-numerical columns based on their cardinality.
        """
        encoded_dataframes = []
        one_hot_encoded_columns = []
        frequency_encoded_columns = []
        hashed_columns = []

        for column in self.non_numerical_columns:
            cardinality = self.data[column].nunique() / len(self.data)
            print(f"Cardinality of column '{column}': {cardinality * 100:.2f}%")

            if cardinality <= 0.05:  # Low cardinality: One-Hot Encoding
                print(f"Applying One-Hot Encoding to column '{column}'...")
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = pd.DataFrame(encoder.fit_transform(self.data[[column]]), columns=encoder.get_feature_names_out([column]))
                encoded_dataframes.append(encoded)
                one_hot_encoded_columns.append(column)
            elif 0.05 < cardinality <= 0.8:  # Medium cardinality: Frequency Encoding
                print(f"Applying Frequency Encoding to column '{column}'...")
                freq_encoding = self.data[column].value_counts(normalize=True)
                self.data[column] = self.data[column].map(freq_encoding)
                frequency_encoded_columns.append(column)
            else:  # High cardinality: Hashing
                print(f"Applying Hashing to column '{column}'...")
                self.data[column] = self.data[column].apply(lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 10)
                hashed_columns.append(column)

        # Combine all encoded dataframes with the original numerical columns
        if encoded_dataframes:
            encoded_data = pd.concat(encoded_dataframes, axis=1)
            self.data = pd.concat([self.data[self.numerical_columns], encoded_data], axis=1)
        else:
            self.data = self.data[self.numerical_columns]

        # Print the columns categorized by encoding method
        print(f"One-Hot Encoded Columns: {one_hot_encoded_columns}")
        print(f"Frequency Encoded Columns: {frequency_encoded_columns}")
        print(f"Hashed Columns: {hashed_columns}")

        # Combine all encoded dataframes with the original numerical columns
        if encoded_dataframes:
            encoded_data = pd.concat(encoded_dataframes, axis=1)
            self.data = pd.concat([self.data[self.numerical_columns], encoded_data], axis=1)
        else:
            self.data = self.data[self.numerical_columns]

    def preprocess(self, columns_to_drop=None, missing_value_threshold=0.5, imputation_strategy='mean', knn_neighbors=5):
        """
        Perform the full preprocessing pipeline.
        """
        self.drop_columns_and_handle_missing(columns_to_drop, missing_value_threshold)
        self.determine_column_types()
        self.impute_missing_values(imputation_strategy, knn_neighbors)
        self.encode_non_numerical_columns()
        print("Preprocessing complete!")
        return self.data


if __name__ == "__main__":
    # Define the path to the dataset
    dataset_path = "../data/dataset/sample_10000.csv"

    # Specify columns to drop
    columns_to_drop = ["code", "url", "creator", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags", "brands_tags", "categories_tags", "categories_fr", "origins_tags", "manufacturing_places", "manufacturing_places_tags", "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags", "first_packaging_code_geo", "cities", "cities_tags", "purchase_places", "countries_tags", "countries_fr", "image_ingredients_url", "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url", "image_small_url", "image_url", "last_updated_t", "last_updated_datetime", "last_modified_by"]

    # Initialize the preprocessor
    preprocessor = DataPreprocessor(file_path=dataset_path)

    # Preprocess the data
    preprocessed_data = preprocessor.preprocess(
        columns_to_drop=columns_to_drop,
        missing_value_threshold=0.3,  # Drop columns with more than 30% missing values
        imputation_strategy='mean',  # Choose 'mean', 'median', 'most_frequent', or 'knn' for imputation
        knn_neighbors=5  # Number of neighbors for KNN Imputer
    )

    # Save the preprocessed data to a new CSV file
    preprocessed_data.to_csv("../data/processed/preprocessed_sample_10000.csv", index=False)
    print("Preprocessed data saved to '../data/processed/preprocessed_sample_10000.csv'.")

    preprocessed_data.info()
