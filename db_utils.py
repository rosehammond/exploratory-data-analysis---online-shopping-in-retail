#%%
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import normaltest
from scipy.stats import yeojohnson
from scipy import stats
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import plotly.express as px
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

    def list_column_skewness(self):

        print("Skewed values:")
        numeric_columns = self.df.select_dtypes(include=[int, float]).columns
        skewness_info = self.df[numeric_columns].skew()
        return skewness_info

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

    def __init__(self, df):
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
        
    def impute_missing_with_mean_median_or_mode(self, column_name, method='mean'):
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
            elif method == 'mode':
                impute_value = self.df[column_name].mode()[0]
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

    def log_transform_skew(self, column_name):
        try:
            self.df[column_name] = self.df[column_name].map(lambda i: np.log(i) if i > 0 else 0)
            t=sns.histplot(self.df[column_name],label="Skewness: %.2f"%(self.df[column_name].skew()) )
            t.legend()
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def yeo_johnson_transform_skew(self, column_name):
        try:
            yeojohnson_column = self.df[column_name]
            yeojohnson_column = stats.yeojohnson(yeojohnson_column)
            yeojohnson_column= pd.Series(yeojohnson_column[0])
            t=sns.histplot(yeojohnson_column,label="Skewness: %.2f"%(yeojohnson_column.skew()) )
            t.legend()
            self.df[column_name] = yeojohnson_column
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def remove_outliers_outside_bounds(self, column_name, lower_bound, upper_bound, inplace=True):
        """
        Remove outliers outside specified upper and lower bounds in a specified column.

        Parameters:
            column_name (str): The name of the column to remove outliers from.
            lower_bound: The lower bound for the valid range.
            upper_bound: The upper bound for the valid range.

        Returns:
            pd.DataFrame: The DataFrame with outliers removed.
        """
        try:
            # Filter the DataFrame to keep rows within the specified bounds
            self.df = self.df[(self.df[column_name] >= lower_bound) & (self.df[column_name] <= upper_bound)]
            return self.df
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")
            return self.df
        
# %%    
# Example usage:
transform_data = DataTransform(customer_df)
transform_data.null_value_summary()
# %%
# Drop rows with data missing in 'operating systems'
transform_data.drop_rows_with_missing_values('operating_systems')
# %%
# Convert object datatypes to category
transform_data = DataTransform(customer_df)
transform_data.convert_column_datatype(['month', 'visitor_type', 'operating_systems','region', 'browser', 'traffic_type', 'administrative', 'informational', 'product_related'], 'category')
#%%
# Check operating systems now contains no non-null values
transform_data.null_value_summary()
# %%
# Check objects have been converted to categories
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
            skewed_value = self.df[column_name].skew()
            print(f"Skew of {column_name} is: {skewed_value}")
            if skewed_value > 2:
                print("This indicates a strong positive skew.")
            elif skewed_value < (-2):
                print("This indicates a strong negative skew.")
            elif skewed_value > 1:
                print("This indicates a moderate positive skew.")
            elif skewed_value < (-1):
                print("This indicates a moderate negative skew.")
            else:
                print("This indicates the data is not significantly skewed.")
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

    def box_plot(self, column_name):
        try:
            fig = px.box(self.df, y=column_name, width=600, height=500)
            fig.show()
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def discrete_probability_distribution(self, column_name):
        # Shows the probabilities of each possible outcome e.g. shows which is most likely to be chosen 
        try:
            plt.rc("axes.spines", top=False, right=False)
            # Calculate value counts and convert to probabilities
            probs = self.df[column_name].value_counts(normalize=True)
            # Create bar plot
            dpd=sns.barplot(y=probs.index, x=probs.values, color='b')
            plt.xlabel('Values')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution: {column_name}')
            plt.show()
        except KeyError:
            print(f"Column '{column_name}' not found in the DataFrame.")

    def correlation_heatmap(self):
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64', 'bool']).columns
        selected_df = self.df[numeric_columns]
        fig = px.imshow(selected_df.corr(), title="Correlation heatmap")
        fig.show()

    def scatter_plot(self, column_1, column_2):
        # Select the two columns you want to plot
        x = self.df[column_1]
        y = self.df[column_2]
        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c='b', marker='o', label='Scatter Plot')
        plt.xlabel(column_1)
        plt.ylabel(column_2)
        plt.title(f'Scatter Plot of {column_1} vs {column_2}')
        plt.legend()
        plt.grid(True)
        plt.show()

# %%
#Example usage:
plot_data = Plotter(customer_df)
#%%
plot_data.bar_plot()
plot_data.missing_data_heatmap()
# %%
# Plot a histogram and a q-q plot for 'product_related_duration'
plot_data.histogram('product_related_duration', 100)
plot_data.quantile_quantile_plot('product_related_duration')
#%%
# Impute missing values from 'product_related_duration' with median
transform_data.impute_missing_with_mean_median_or_mode('product_related_duration', 'median')
#%%
# Drop column 'administrative_duration' 
transform_data = DataTransform(customer_df)
transform_data.drop_column('administrative_duration')
transform_data.drop_column('administrative')
#%%
# Check columns are no longer in DataFrame
customer_df.info()
#%%
# Impute missing values from administrative and informational_duration with median
transform_data = DataTransform(customer_df)
basic_df_info.column_stats('informational_duration')
transform_data.impute_missing_with_mean_median_or_mode('informational_duration', 'median')
#%%
# Find information about 'product_related' to decide how to impute
basic_df_info.count_unique_values('product_related')
# %%
# Impute missing values with the mode as the column is now categorical
transform_data = DataTransform(customer_df)
transform_data.impute_missing_with_mean_median_or_mode('product_related', 'mode')
# %%
# Heatmap now shows no values are missing
plot_data.missing_data_heatmap()
#%%
#Display the skewness for each column
basic_df_info.list_column_skewness()
# %%
# Checking skewness with value and histogram for particular columns:
plot_data.histogram('informational_duration')
# %%
# View some informational data
customer_df['informational_duration'].describe()
plot_data.quantile_quantile_plot('informational_duration')
# %%
# Transform the 'informational_duration' column to reduce skewness
transform_data.log_transform_skew('informational_duration')
# %%
# Transform the 'exit_rates' column
transform_data.yeo_johnson_transform_skew('exit_rates')
transform_data.yeo_johnson_transform_skew('bounce_rates')
transform_data.yeo_johnson_transform_skew('product_related_duration')
transform_data.yeo_johnson_transform_skew('page_values')
# %%
# Check the data has been changed in the original dataframe
basic_df_info.list_column_skewness()
# %%
# Used below code to check transformations before making them permanent in the dataframe
log_population = customer_df["page_values"].map(lambda i: np.log(i) if i > 0 else 0)
t=sns.histplot(log_population,label="Skewness: %.2f"%(log_population.skew()) )
t.legend()
# %%
yeojohnson_population = customer_df["page_values"]
yeojohnson_population = stats.yeojohnson(yeojohnson_population)
yeojohnson_population= pd.Series(yeojohnson_population[0])
t=sns.histplot(yeojohnson_population,label="Skewness: %.2f"%(yeojohnson_population.skew()) )
t.legend()
# %%
# Drop rows that now have missing values (only 9 rows)
transform_data.drop_rows_with_missing_values('exit_rates')
#%%
# Save the transformed data as a new file
save_dataframe_to_csv(customer_df, "transformed_customer_activity_data.csv")
# %%
customer_df.info()
# %%
# Create new dataframe with updated data
transformed_customer_df = load_data_from_csv("transformed_customer_activity_data.csv")
# %%
transformed_customer_df.info()
# %%
# Create an instance of the DataTransform class with updated dataframe
transform_updated_data = DataTransform(transformed_customer_df)
#%%
# Convert some datatypes to category
transform_updated_data.convert_column_datatype(['month', 'visitor_type', 'operating_systems','region', 'browser', 'traffic_type', 'informational', 'product_related'], 'category')
# %%
# Create an instance of the DataFrameInfo class with updated dataframe
basic_updated_df_info = DataFrameInfo(transformed_customer_df)
basic_updated_df_info.basic_summary(6)
# %%
# Create an instance of the Plotter class with updated dataframe
plot_updated_data = Plotter(transformed_customer_df)
# %%
# Create box plots to check for outliers in updated dataframe
plot_updated_data.box_plot("informational_duration")
# %%
# Create discrete probability distribution to see popular variables
plot_updated_data.discrete_probability_distribution('month')
# %%
plot_updated_data.quantile_quantile_plot('exit_rates')
# %%
transformed_customer_df = transform_updated_data.remove_outliers_outside_bounds('informational_duration', -0.1000000, 8.0000000)
# %%
plot_updated_data = Plotter(transformed_customer_df)
plot_updated_data.box_plot("informational_duration")
# %%
transformed_customer_df = transform_updated_data.remove_outliers_outside_bounds('product_related_duration', -1.0, 36.0)
# %%
plot_updated_data = Plotter(transformed_customer_df)
plot_updated_data.box_plot("product_related_duration")
# %%
transformed_customer_df.info()
# %%
numeric_columns = transformed_customer_df.select_dtypes(include=['int64', 'float64', 'bool']).columns
numeric_columns_transformed_customer_df = transformed_customer_df[numeric_columns]
px.imshow(numeric_columns_transformed_customer_df.corr(), title="Correlation heatmap of customer dataframe")
# %%
transform_updated_data = DataTransform(transformed_customer_df)
transform_updated_data.convert_column_datatype(['informational', 'product_related'], 'int64')
# %%
# PLot a heatmap to show the correlation values for numeric columns
plot_updated_data = Plotter(transformed_customer_df)
plot_updated_data.correlation_heatmap()
# %%
# PLot a scatter plot for columns with a correlation value > 0.7
plot_updated_data = Plotter(transformed_customer_df)
plot_updated_data.scatter_plot('informational_duration', 'informational')
# %%
plot_updated_data.scatter_plot('bounce_rates', 'exit_rates')
# %%
transform_updated_data.drop_column('informational_duration')
# %%
# Check the column has been removed
transformed_customer_df.info()
# %%
