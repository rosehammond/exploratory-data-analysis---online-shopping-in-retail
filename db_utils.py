#%%
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import normaltest
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import seaborn as sns
import yaml
#%%
def yaml_to_dict(filename="credentials.yaml"): 
    # Function takes in a filename as a paramenter, will default to 
    # 'credentials.yaml' if not given
    try:
        with open(filename, 'r') as file:
            data_loaded = yaml.safe_load(file)
        return data_loaded
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing '{filename}': {e}")
        return None

class RDSDatabaseConnector:
    
    def __init__(self, credentials_data):
        self.host = credentials_data.get("RDS_HOST")
        self.port = credentials_data.get("RDS_PORT")
        self.database = credentials_data.get("RDS_DATABASE")
        self.user = credentials_data.get("RDS_USER")
        self.password = credentials_data.get("RDS_PASSWORD")

        # Initialize connection and cursor as None
        self.connection = None
        self.cursor = None

    def open_connection(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor()
            print("Database connection established successfully.")
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.close_connection()  # Close connection if there's an error

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def create_engine(self):
        if not hasattr(self, 'engine') or self.engine is None:
            try:
                # Construct the database URL for SQLAlchemy
                db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
                
                # Create the SQLAlchemy engine
                self.engine = create_engine(db_url)
                print("SQLAlchemy engine created successfully.")
            except Exception as e:
                print(f"Error creating SQLAlchemy engine: {e}")

    def execute_query(self, query):
        try:
            result = pd.read_sql(query, self.engine)
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def extract_customer_activity(self):
        try:
            # Define a SQL query to select data from the "customer_activity" table
            query = "SELECT * FROM customer_activity"
            # Use the execute_query method to retrieve data
            result_df = self.execute_query(query)
            return result_df
        except Exception as e:
            print(f"Error extracting customer activity data: {e}")
            return None

def save_dataframe_to_csv(dataframe, filename):
    try:
        dataframe.to_csv(filename, index=False)
        print(f"Data saved to {filename} successfully.")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")
# %%
#Load credentials from the file
credentials_data = yaml_to_dict("credentials.yaml")  
if credentials_data:
    db_connector = RDSDatabaseConnector(credentials_data)  # Initialize the connector
    db_connector.open_connection()  # Open the database connection
    db_connector.create_engine()  # Initialise the SQLAlchemy engine

    if db_connector:
        customer_activity_data = db_connector.extract_customer_activity()

        if customer_activity_data is not None:
            save_dataframe_to_csv(customer_activity_data, "customer_activity_data.csv")

    if db_connector:
        db_connector.close_connection()  # Close the database connection

# %%
def load_data_from_csv(filename):
    try:
        dataframe = pd.read_csv(filename)
        return dataframe
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None

if __name__ == "__main__":
    customer_df = load_data_from_csv("customer_activity_data.csv")
    if customer_df is not None:
        # Print the shape of the DataFrame
        print("Data shape:", customer_df.shape)
        # Print the first few rows of the DataFrame
        print("\nSample of the data:")
        print(customer_df.head())
# %%
class DataFrameInfo:

    def __init__(self, df=customer_df):
        self.df = df

    def basic_summary(self, number_of_rows):
        print(f"First {number_of_rows} rows:")
        print(self.df.head(number_of_rows)) 
        print(f"Last {number_of_rows} rows:")
        print(self.df.tail(number_of_rows))

    def get_basic_info(self):
        print("Data shape:") 
        print(self.df.shape)
        print("Data info:") 
        print(self.df.info())
        print("Data stats:") 
        print(self.df.describe())
    
    def count_unique_values(self, column_name):
        # Extract the specified column
        column = self.df[column_name]
        # Count the unique values
        unique_values_count = column.value_counts().reset_index()
        # Rename the columns
        unique_values_count.columns = [column_name, 'Count']
        
        return unique_values_count
    
    def column_stats(self, column_name):
       
        try:
            column = self.df[column_name]
            if pd.api.types.is_numeric_dtype(column):
                print(f"The stats for {column_name} are:")
                print(f"mean: {column.mean()}, median: {column.median()}, mode: {column.mode()[0]}, std: {column.std()}")
            else:
                return "Column is not numeric"
        except KeyError:
            return f"Column '{column_name}' not found in the DataFrame"
        
    def column_null_stats(self, column_name): 
        
        try:
            column = self.df[column_name]
            null_count = column.isnull().sum()
            total_count = len(column)
            if total_count > 0:
                null_percentage = (null_count / total_count) * 100
                print(f"Column '{column_name}':")
                print(f"Number of Nulls: {null_count}")
                print(f"Percentage of Nulls: {null_percentage:.2f}%")
            else:
                print(f"Column '{column_name}' is empty.")
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

# %%
#Example usage to find out information (does not alter the data):
basic_df_info = DataFrameInfo(customer_df)
#%%
basic_df_info.get_basic_info()
basic_df_info.count_unique_values('region')
basic_df_info.count_unique_values('month')
basic_df_info.column_stats('product_related_duration')
basic_df_info.column_null_stats('administrative') 
basic_df_info.basic_summary(7)
# %%  
basic_df_info.column_null_stats('product_related') 
# %%   

class DataTransform:

    def __init__(self, df=customer_df):
        self.df = df

    def convert_column_datatype(self, column_name, new_datatype):
        try:
            self.df[column_name] = self.df[column_name].astype(new_datatype)
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")
        except ValueError:
            print(f"Failed to change the datatype of column '{column_name}' to {new_datatype}.")
        return self.df
    
    def null_value_summary(self):
       
        # Calculate the count of null values in each column
        null_counts = self.df.isnull().sum()
        # Calculate the percentage of null values in each column
        total_rows = len(self.df)
        null_percentages = (null_counts / total_rows) * 100
        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Column Name': null_counts.index,
            'Null Count': null_counts.values,
            'Null Percentage (%)': null_percentages.values
        })
        # Filter out columns with no null values
        summary_df = summary_df[summary_df['Null Count'] > 0]

        return summary_df
    
    def drop_column(self, column_name):
        """
        Drop a specified column from a DataFrame.

        Parameters:
            data (pd.DataFrame): The DataFrame from which to drop the column.
            column_name (str): The name of the column to be dropped.

        Returns:
            pd.DataFrame: The DataFrame with the specified column removed.
        """
        try:
            self.df = self.df.drop(column_name, axis=1, inplace=True)
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")
        
    def impute_missing_with_mean_or_median(self, column_name, method='mean'):
        """
        Impute missing values in a specified column with the mean or median.

        Parameters:
            column_name (str): The name of the column to impute.
            method (str): The imputation method, either 'mean' or 'median' (default is 'mean').

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed.
        """
        try:
            if method == 'mean':
                impute_value = self.df[column_name].mean()
            elif method == 'median':
                impute_value = self.df[column_name].median()
            else:
                raise ValueError("Invalid imputation method. Use 'mean' or 'median'.")
            
            self.df[column_name].fillna(impute_value, inplace=True)
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")
    
    def drop_rows_with_missing_values(self, column_name):
        """
        Drop rows that have missing values in a specific column.

        Parameters:
            column_name (str): The name of the column to check for missing values.

        Returns:
            pd.DataFrame: The DataFrame with rows containing missing values in the specified column removed.
        """
        try:
            self.df = self.df.dropna(subset=[column_name], inplace=True)
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

# %%    
# Example usage:
transform_data = DataTransform(customer_df)
transform_data.null_value_summary()
# %%
# Drop rows with data missing in 'operating systems'
transform_data.drop_rows_with_missing_values('operating_systems')
# %%
# Convert object datatypes to category
transform_data.convert_column_datatype(['month', 'visitor_type', 'operating_systems','region', 'browser', 'traffic_type'], 'category')
#%%
transform_data.null_value_summary()
# %%
customer_df.info()
# %%

class Plotter:
    
    def __init__(self, df=customer_df):
        self.df = df

    def bar_plot(self):
        # Calculate the percentage of missing values in each column
        total_rows = len(self.df)
        null_percentages = (self.df.isnull().sum() / total_rows) * 100

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(null_percentages.index, null_percentages.values)
        plt.xlabel('Percentage of Missing Values')
        plt.ylabel('Columns')
        plt.title('Percentage of Missing Values in Each Column')
        plt.gca().invert_yaxis()  # Invert the y-axis for better readability
        plt.show()

    def normality_test(self, column_name):
        # D’Agostino’s K^2 Test
        # a p-value of less than 0.05 provides significant evidence for the 
        # null hypothesis being false
        try:
            data = self.df[column_name]
            stat, p = normaltest(data, nan_policy='omit')
            print('Statistics=%.3f, p=%.3f' % (stat, p))
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def histogram(self, column_name, num_bins=50):
        #number of bins defaults ot 50 if not specified
        try:
            print(f"Histogram for {column_name}:")
            self.df[column_name].hist(bins=num_bins)
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def quantile_quantile_plot(self, column_name):
        try:
            print(f"Q-Q plot for {column_name}:")
            qq_plot = qqplot(self.df[column_name] , scale=1 ,line='q')
            plt.show()
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def missing_data_heatmap(self):
        # Create a missing data matrix (heatmap)
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.title('Missing Data Matrix (Heatmap)')
        plt.show()

# %%
#Example usage:
plot_data = Plotter(customer_df)
plot_data.bar_plot()
plot_data.missing_data_heatmap()
# %%
# Plot a histogram and a q-q plot for 'product_related_duration'
plot_data.histogram('product_related_duration', 100)
plot_data.quantile_quantile_plot('product_related_duration')
#%%
# Impute missing values from 'product_related_duration' with median
transform_data.impute_missing_with_mean_or_median('product_related_duration', 'median')
#%%
# Drop column 'administrative_duration' 
transform_data.drop_column('administrative_duration')
#%%
# Impute missing values from administrative and informational_duration with median
basic_df_info.column_stats('administrative')
basic_df_info.column_stats('informational_duration')
transform_data.impute_missing_with_mean_or_median('administrative', 'median')
transform_data.impute_missing_with_mean_or_median('informational_duration', 'median')
#%%
# Find information about 'product_related' to decide how to impute
basic_df_info.count_unique_values('product_related')
plot_data.quantile_quantile_plot('product_related')
plot_data.histogram('product_related')
# %%
# Impute missing values with the median as the mean gives a float and all other valus are int
transform_data.impute_missing_with_mean_or_median('product_related', 'median')
# %%
# Heatmap now shows no values are missing
plot_data.missing_data_heatmap()
#%%