#%%
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import yaml

#create a function which loads the credentials.yaml 
# file and returns the data dictionary contained within. 
# This will be be passed to your RDSDatabaseConnector as an 
# argument which the class will use to connect to the remote database.

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
credentials_data = yaml_to_dict("credentials.yaml")  # Load credentials from your file
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
pd.set_option('display.max_columns', None)

def load_data_from_csv(filename):
    try:
        dataframe = pd.read_csv(filename)
        return dataframe
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None

# Example usage:
loaded_data = load_data_from_csv("customer_activity_data.csv")
if loaded_data is not None:
        # Print the shape of the DataFrame
    print("Data shape:", loaded_data.shape)
    
    # Print the first few rows of the DataFrame
    print("\nSample of the data:")
    print(loaded_data.head())
# %%
loaded_data.info()
loaded_data.describe()

# %%
