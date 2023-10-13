# Exploratory Data Analysis - Online Shopping in Retail


## Table of Contents

- Description
- Technologies utilised
- Installation instructions
- Usage instructions
- File structure of the project
- Code structure
- License information


## Description (what it does, the aim of the project, and what you learned)

Brief: You currently work for a large retail company where you have the task of conducting exploratory data analysis on a dataset of online shopping website activity. With the increasing popularity of online shopping, this dataset provides valuable insights into consumer behaviour.

The project will use various statistical and visualisation techniques to explore the data and draw meaningful insights. The analysis will not only provide a better understanding of customer behaviour but also help the business optimise marketing strategies and improve its customer experience.

The data contains information about the website activity of users over one year. Each sample represents the user interacting with the website during a shopping session.

Overall, this project showcases the power of exploratory data analysis in uncovering valuable insights from large datasets and how to leverage these insights to drive business success.


## Technologies utilised

- Languages:
  - Python
- Libraries:
  - Matplotlib
  - NumPy
  - Pandas
  - Plotly
  - Psycopg2
  - Scipy
  - Seaborn
  - Sqlalchemy
  - Statsmodels
  - Yaml


## Installation instructions

1. Clone the repository to your local machine: git clone https://github.com/rosehammond/exploratory-data-analysis---online-shopping-in-retail.git
2. Navigate to the project directory in VSCode
3. Use the terminal to install the above Python libraries if you don't have them using _pip install (library_name)_


## Usage instructions

1. __Set up credentials:__ Store your database credentials in a YAML file named _credentials.yaml_
2. __Load Data from Database:__ Use the `RDSDatabaseConnector()` class to open a connection, create a SQLAlchemy engine, and execute SQL queries to retrieve data.
3. __Save Data to CSV:__ Extracted data from the database can be saved as a CSV file using the `save_dataframe_to_csv(dataframe, filename)` function. 
4. __Load Data from CSV:__ You can load data from a CSV file using the `load_data_from_csv(filename)` function.
5. __Exploratory Data Analysis:__ Use the `DataFrameInfo()`, `DataTransform()`, and `Plotter()` classes to retrieve information, visualise, and transform the data as you need.


## File structure of the project

- __db_utils.py__ contains all the Python code


## Code structure

- `yaml_to_dict(filename)` Function which takes in a YAML file storing your credentials and converts it to a dictionary.
- `RDSDatabaseConnector(credentials_data)` Class which takes in the credentials dictionary as a parameter and uses methods to connect to the database, extract the information, and close the connection.
- `save_dataframe_to_csv(dataframe, filename)` Function to save the extracted data to a CSV file
- `load_data_from_csv(filename)` Function to load the CSV file to a DataFrame
- `DataFrameInfo(dataframe)` Class which contains methods to retrieve information about the data e.g. `count_unique_values(column_name)' which returns the unique values from a column and their frequency.
- `DataTransform(dataframe)` Class which contains methods to transform the data e.g. to remove outliers, drop columns with missing rows, etc.
- `Plotter(dataframe)` Class which contains methods to visualise the data e.g. bar plot, histogram, correlation heatmap, etc.


## License information

This project is licensed under the MIT License - see the LICENSE.txt file for details.
