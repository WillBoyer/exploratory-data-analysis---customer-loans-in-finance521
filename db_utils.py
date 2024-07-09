import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
import numpy as np
import seaborn as sns
from scipy import stats, special
import plotly.graph_objects as go


def get_credentials(filename):
    '''
    Retrieves the credentials from the 'credentials.yaml' file,
    which can then be used to connect to the database.
    
    Args:
        filename (str): The YAML file that contains the database credentials.
        
    Returns:
        dict: A Python dictionary of the database credentials.
    '''
    with open(filename, 'r') as cred_file:
        creds = yaml.safe_load(cred_file)['connection']
    return creds

class RDSDatabaseConnector:
    '''
    Represents the connection to the database
    
    Attributes:
        creds (dict): The credentials to connect to the database.
    '''
    def __init__(self, credentials):
        '''
        See help(RDSDatabaseConnector) for accurate signature.
        '''
        self.creds = credentials
        
    def create_conn(self):
        '''
        Uses the credentials to connect to the database.
        
        Returns:
            sqlalchemy.engine.base.Engine: The output of the 'create_engine' function.
        '''
        user = self.creds['RDS_USER']
        password = self.creds['RDS_PASSWORD']
        host = self.creds['RDS_HOST']
        port = self.creds['RDS_PORT']
        dbname = self.creds['RDS_DATABASE']
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
        return engine
    
    def get_dataframe(self, engine):
        '''
        Submits a query to the database, and stores the result a
        Pandas dataframe.
        
        Args:
            engine (sqlalchemy.engine.base.Engine): The database query engine.
        
        Returns:
            pandas.DataFrame: The SQL database table, as converted to a Pandas Dataframe.
        '''
        df = pd.read_sql_query('''SELECT * FROM loan_payments''', engine).set_index('id')
        return df
    
    def save_to_file(self, dataframe):
        '''
        Converts the Pandas dataframe into a CSV file.
        
        Args:
            dataframe (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
        '''
        dataframe.to_csv('out.csv', index=False)
    
class DataFrameInfo:
    '''
    This class contains several methods to gain information about the dataframe.
    '''
    def describe(self, df):
        '''
        Lists the data types of all dataframe columns.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
        '''
        for c in df:
            print(f"Column '{c}' has data type {df[c].dtype}")
        
    def get_stats(self, df):
        '''
        Lists statistics for all columns of the dataframe.
        For numerical data, this includes mean, median and standard deviation.
        For all other data types, this is instead the mode of each column.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
        '''
        for c in df:
            print(f"Column '{c}':")
            # Stats will only work on numerical data
            if (df[c].dtype != object):
                col_mean = df[c].mean()
                col_median = df[c].median()
                col_std = df[c].std()
                print(f"\t Mean: {col_mean}")
                print(f"\t Median: {col_median}")
                print(f"\t Standard Deviation: {col_std}")
            else:
                col_mode = df[c].mode().iat[0]
                print(f"\t Mode: {col_mode}")
        
    def count_distinct(self, df):
        '''
        Counts the numhber of unique values in each of the columns that
        represent categorical data.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
        '''
        # Need to only analyse categorical data
        categorical_columns = ['term', 
                               'grade',
                               'sub_grade',
                               'employment_length',
                               'home_ownership',
                               'verification_status',
                               'loan_status',
                               'payment_plan',
                               'purpose',
                               'policy_code',
                               'application_type']
        for c in categorical_columns:
            no_distinct = len(pd.unique(df[c]))
            print(f"Column '{c}' has {no_distinct} unique values")
        
    def print_shape(self, df):
        '''
        Prints the number of rows and columns in the dataframe, in a user-
        readable format.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
        '''
        shape = df.shape
        print(f"The dataframe has {shape[0]} rows and {shape[1]} columns")
    
    def count_null(self, df):
        '''
        Prints the number of null values in every column, both as a count and
        a percentage of the total rows, in a human-readable format.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
        '''
        for c in df:
            column_name = c
            null_count = df[c].isna().sum()
            null_percentage = df[c].isna().mean() * 100
            if null_count != 0:
                print(f"Column '{column_name}' has {null_count} null values ({null_percentage}% of total values)")
                
    def get_skews(self, df):
        '''
        Prints the skew of all numerical columns in the dataframe, in a human-
        readable format.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
        '''
        for column in df:
            if df[column].dtype != 'object' and column != 'member_id':
                print(f"Skew of {column} is {df[column].skew()}")
                
    def get_z_score(self, df, column):
        '''
        Creates a new dataframe for the specified column, containing the
        z-score for each value in the column.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be analysed.
            
        Returns:
            pandas.Dataframe: A dataframe with the input column, plus a column
                showing the z-score for each of the values.
        '''
        col_mean = df[column].mean()
        col_std = df[column].std()
        col_z_scores = (df[column] - col_mean) / col_std
        analysis_df = df[[column]]
        analysis_df['z_scores'] = col_z_scores
        return analysis_df
    
    def get_interquartile_range(self, df, column):
        '''
        Calculates the Interquartile Range (IQR) for a given column, prints it
        in a user-readable format, and cuts down the dataframe to exclude all 
        values which are either:
            - Lower than Q1 by 1.5 * IQR
            - Higher than Q3 by 1.5 * IQR
        
        The IQR is the range from the 1st quartile (Q1) to the 3rd quartile 
        (Q3). If a value lies outside the range specified above, it is counted
        as an outlier.
        
        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be analysed.
            
        Returns:
            pandas.Dataframe: A cut-down dataframe, containing only the rows 
                which contain outliers in the current column.
        '''
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        
        IQR = Q3 - Q1
        print(f"Q1 (25th percentile): {Q1}")
        print(f"Q3 (75th percentile): {Q3}")
        print(f"IQR: {IQR}")

        # Identify outliers
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        return outliers
    
    def get_vif(self, df, column):
        return

class DataTransform:
    None

class Plotter:
    '''
    Produces plots to visualise aspects of the dataset such as skew.
    '''
    @classmethod
    def show_all_skews(self, df, columns):
        '''
        For a given list of columns, produce histogram plots to visualise skew.
        
        Args:
            columns (string[]): A list of column names to be analysed for skew.
        '''
        sns.set(font_scale=0.7)
        f = pd.melt(df, value_vars=columns)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
            
    @classmethod
    def plot_pie_category(self, df, column):
        '''
        For a given column, produce a pie chart showing the percentage of rows
        which have each value.
        
        Args:
            column (string): The name of the column to be analysed.
            
        Returns:
            plotly.graph_objects.Pie: The pie chart object.
        '''
        categories = sorted([cat for cat in df[column].unique()])
        
        values = []        
        
        for c in categories:
            count = len(df[df[column].str.contains(c)])
            values.append(count)
        
        category_pie = go.Pie(values=values, 
                              labels=categories)
        return category_pie        
    
class DataFrameTransform:
    '''
    Contains methods to perform transforms on the dataset.
    '''
    @classmethod
    def impute_with_median(self, df, column):
        '''
        Replaces NULL values with the median value of the column.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be transformed.

        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        median = df[column].median()
        df.fillna({column: median}, inplace=True)
        return df
    
    @classmethod
    def impute_with_mean(self, df, column):
        '''
        Replaces NULL values with the mean value of the column.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be transformed.

        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        mean = df[column].mean()
        df.fillna({column: mean}, inplace=True)
        return df

    @classmethod
    def impute_with_mode(self, df, column):
        '''
        Replaces NULL values with the most common value of the column.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be transformed.

        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        mode = df[column].mode()[0]
        df.fillna({column: mode}, inplace=True)
        return df

    @classmethod
    def drop_null_rows(self, df, column):
        '''
        Drops all rows from the dataframe which have a NULL value in the given 
        column.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column containing the NULL values.

        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        df.dropna(axis=0, how='any', subset = column, inplace=True)
        return df
    
    @classmethod
    def log_transform(self, df, column):
        '''
        Performs a Log transform to correct skew in the given column of the
        dataframe.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be transformed.

        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        log_output = df[column]
        log_output = log_output.map(lambda i: np.log(i) if i > 0 else 0)
        return log_output
    
    @classmethod
    def box_cox_transform(self, df, column):
        '''
        Performs a Log transform to correct skew in the given column of the
        dataframe.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be transformed.

        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        boxcox_output = df[column]
        boxcox_output = boxcox_output.map(lambda i: special.boxcox1p(i, 0.25))
        return boxcox_output
    
    @classmethod
    def yeo_johnson_transform(self, df, column):
        '''
        Performs a Yeo-Johnson transform to correct skew in the given column of
        the dataframe.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column to be transformed.

        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        yeo_johnson_output = df[column]
        yeo_johnson_output = yeo_johnson_output.map(lambda i: stats.yeojohnson(i))
        return yeo_johnson_output
    
    @classmethod
    def drop_outliers(self, df, column):
        '''
        Drops values from a dataframe column with a z-score of 3 or higher.

        Args:
            df (pandas.Dataframe): The SQL database table, as converted to a Pandas Dataframe.
            column (string): The name of the column containing the outliers.
        
        Returns:
            pandas.Dataframe: The transformed dataframe.
        '''
        df = df[(np.abs(stats.zscore(df[column])) < 3)]
        return df

if __name__ == "__main__":
    creds = get_credentials('credentials.yaml')
    RDSDBConn = RDSDatabaseConnector(creds)
    engine = RDSDBConn.create_conn()
    df = RDSDBConn.get_dataframe(engine)
    # RDSDBConn.save_to_file(df)
    
    dfInfo = DataFrameInfo()
    dfInfo.describe(df)
    dfInfo.get_stats(df)
    dfInfo.count_distinct(df)
    dfInfo.print_shape(df)
    dfInfo.count_null(df)
    
    
    # Handling Null Values
    # Dropping columns. Personally, would keep 'next_payment_date', but Project
    # instructions require there to be no null values at the end
    df = df.drop(['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog', 'next_payment_date'], axis=1)
    # Imputation
    dfTransform = DataFrameTransform()
    df = dfTransform.drop_null_rows(df, ['last_payment_date', 'last_credit_pull_date', 'collections_12_mths_ex_med'])
    df = dfTransform.impute_with_mode(df, 'term')
    df = dfTransform.impute_with_mode(df, 'employment_length')
    df = dfTransform.impute_with_mean(df, 'funded_amount')
    df = dfTransform.impute_with_mean(df, 'int_rate')
    
    # Verify that no more null values are required
    dfInfo.count_null(df)
    
    numeric_columns = [
        'loan_amount',
        'funded_amount',
        'funded_amount_inv',
        'int_rate',
        'instalment',
        'annual_inc',
        'dti',
        'inq_last_6mths',
        'open_accounts',
        'total_accounts',
        'out_prncp',
        'out_prncp_inv',
        'total_payment',
        'total_payment_inv',
        'total_rec_prncp',
        'total_rec_int',
        'total_rec_late_fee',
        'recoveries',
        'collection_recovery_fee',
        'last_payment_amount'
    ]
    
    for column in numeric_columns:
        if df[column].dtype != 'object' and column != 'member_id':
            print(f"Skew of {column} is {df[column].skew()}")
            
    for c in numeric_columns:
        col_skew = df[c].skew()
        if col_skew > 1 or col_skew < -1:
            df[c] = DataFrameTransform.log_transform(df, c)
            
    for column in numeric_columns:
        if df[column].dtype != 'object' and column != 'member_id':
            print(f"Skew of {column} is {df[column].skew()}")