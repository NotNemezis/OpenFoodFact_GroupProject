import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataLoader:
    def __init__(self, file_path, file_type='csv'):
        """
        Initialize the DataLoader with the dataset path and parameters.

        :param file_path: Path to the dataset file.
        :param file_type: Type of the file ('csv', 'json', 'excel'). Default is 'csv'.
        :param chunk_size: Size of chunks to load the dataset in parts (useful for large datasets). Default is None.
        """
        self.file_path = file_path
        self.file_type = file_type

    def load_data(self):
        """
        Load the dataset based on the file type and parameters.

        :return: Loaded dataset as a pandas DataFrame or a generator if chunk_size is specified.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file at {self.file_path} does not exist.")

        if self.file_type == 'csv':
            """
            Error description:
            Columns (11,17,33,35,72) have mixed data types. Specify dtype option on import or set low_memory=False.
            df = pd.read_csv(self.file_path, nrows=100000, sep='\t', encoding='utf-8', low_memory=True, na_filter=True)# Select the problematic columns by their index positions
            columns_to_check = df.columns[[11, 17, 33, 35, 72]] allows you to select the problematic columns by their index positions.
            """
            #logging.info("Dataset comes from a csv file!")
            # df = pd.read_csv(self.file_path, nrows=100000, sep='\t', encoding='utf-8', low_memory=True, na_filter=True)# Select the problematic columns by their index positions
            # columns_to_check = df.columns[[11, 17, 33, 35, 72]]  # Get column names from indices

            # # Print the data types of these columns
            # print(df[columns_to_check].dtypes)
            return pd.read_csv(self.file_path, nrows=100000, sep='\t', encoding='utf-8', low_memory=True, na_filter=True)
        elif self.file_type == 'json':
            return pd.read_json(self.file_path)
        elif self.file_type == 'excel':
            return pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def save_data(self, data, save_path, file_type='csv'):
        """
        Save the given DataFrame to the specified file path.

        :param data: The pandas DataFrame to save.
        :param save_path: The path where the file will be saved.
        :param file_type: The type of file to save ('csv', 'json', 'excel'). Default is 'csv'.
        """
        logging.info(f"Saving data to {save_path}...")
        if file_type == 'csv':
            data.to_csv(save_path, index=False)
        elif file_type == 'json':
            data.to_json(save_path, orient='records', lines=True)
        elif file_type == 'excel':
            data.to_excel(save_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        logging.info(f"Data saved successfully to {save_path}!")
        
    def run(self, sample_size, save_path, save_file_type='csv'):
        """
        Run the full workflow: load data, sample it, and save the sampled data.

        :param sample_size: Number of rows to sample from the dataset.
        :param save_path: The path where the sampled data will be saved.
        :param save_file_type: The type of file to save ('csv', 'json', 'excel'). Default is 'csv'.
        """
        try:
            # Load the data
            data = self.load_data()
            logging.info("Dataset loaded successfully!")

            # Sample the data
            logging.info(f"Sampling {sample_size} rows from the dataset...")
            sampled_data = data.sample(n=sample_size, random_state=42)
            logging.info(f"Sampled {sample_size} rows successfully!")

            # Save the sampled data
            self.save_data(sampled_data, save_path, file_type=save_file_type)

        except Exception as e:
            print(f"Error during the run process: {e}")

# Example usage
# if __name__ == "__main__":
#     # Define the path to your dataset
#     dataset_path = "../data/dataset/en.openfoodfacts.org.products.csv"

#     # Initialize the DataLoader
#     data_loader = DataLoader(file_path=dataset_path, file_type='csv')

#     # Load the data
#     try:
#         data = data_loader.load_data()
#         print("Dataset loaded successfully!")

#         # Retrieve 100,000 random items
#         sampled_data = data.sample(n=10000, random_state=42)

#         # Save 10,000 items from the sample to a new CSV file
#         sampled_data.to_csv("../data/dataset/sample_10000.csv", index=False)
#         print("Sample of 10,000 items saved to 'sample_10000.csv'.")
        
#     except Exception as e:
#         print(f"Error loading data: {e}")