import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
import numpy as np
import seaborn as sns
from scipy import stats, special


def get_credentials(file):
    with open('credentials.yaml', 'r') as cred_file:
        creds = yaml.safe_load(cred_file)['connection']
    return creds

class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.creds = credentials
        
    def create_conn(self):
        user = self.creds['RDS_USER']
        password = self.creds['RDS_PASSWORD']
        host = self.creds['RDS_HOST']
        port = self.creds['RDS_PORT']
        dbname = self.creds['RDS_DATABASE']
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
        return engine
    
    def get_dataframe(self, engine):
        df = pd.read_sql_query('''SELECT * FROM loan_payments''', engine).set_index('id')
        return df
    
    def save_to_file(self, dataframe):
        dataframe.to_csv('out.csv', index=False)
    
class DataFrameInfo:
    def describe(self, df):
        for c in df:
            print(f"Column '{c}' has data type {df[c].dtype}")
        
    def get_stats(self, df):
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
        shape = df.shape
        print(f"The dataframe has {shape[0]} rows and {shape[1]} columns")
    
    def count_null(self, df):
        for c in df:
            column_name = c
            null_count = df[c].isna().sum()
            null_percentage = df[c].isna().mean() * 100
            if null_count != 0:
                print(f"Column '{column_name}' has {null_count} null values ({null_percentage}% of total values)")
                
    def get_skews(self, df):
        for column in df:
            if df[column].dtype != 'object' and column != 'member_id':
                print(f"Skew of {column} is {df[column].skew()}")
                
    def get_z_score(self, df, column):
        col_mean = df[column].mean()
        col_std = df[column].std()
        col_z_scores = (df[column] - col_mean) / col_std
        analysis_df = df[[column]]
        analysis_df['z_scores'] = col_z_scores
        return analysis_df
    
    def get_interquartile_range(self, df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        
        IQR = Q3 - Q1
        print(f"Q1 (25th percentile): {Q1}")
        print(f"Q3 (75th percentile): {Q3}")
        print(f"IQR: {IQR}")

        # Identify outliers
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        print("Outliers:")
        print(outliers)
        return outliers

class DataTransform:
    None

class Plotter:
    @classmethod
    def show_all_skews(self, df, columns):
        sns.set(font_scale=0.7)
        f = pd.melt(df, value_vars=columns)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
    
class DataFrameTransform:
    @classmethod
    def impute_with_median(self, df, column):
        median = df[column].median()
        # df[column].fillna(median, inplace=True)
        df.fillna({column: median}, inplace=True)
        return df
    
    @classmethod
    def impute_with_mean(self, df, column):
        mean = df[column].mean()
        # df[column].fillna(mean, inplace=True)
        df.fillna({column: mean}, inplace=True)
        return df

    @classmethod
    def impute_with_mode(self, df, column):
        mode = df[column].mode()[0]
        # df[column].fillna(mode, inplace=True)
        df.fillna({column: mode}, inplace=True)
        return df

    @classmethod
    def drop_null_rows(self, df, column):
        df.dropna(axis=0, how='any', subset = column, inplace=True)
        return df
    
    @classmethod
    def log_transform(self, df, column):
        log_output = df[column]
        log_output = log_output.map(lambda i: np.log(i) if i > 0 else 0)
        return log_output
    
    @classmethod
    def box_cox_transform(self, df, column):
        boxcox_output = df[column]
        # boxcox_output = boxcox_output.map(lambda i: stats.boxcox(i)[0])
        boxcox_output = boxcox_output.map(lambda i: special.boxcox1p(i, 0.25))
        return boxcox_output
    
    @classmethod
    def yeo_johnson_transform(self, df, column):
        yeo_johnson_output = df[column]
        # yeo_johnson_output = yeo_johnson_output.map(lambda i: np.log(i) if i > 0 else 0)
        yeo_johnson_output = yeo_johnson_output.map(lambda i: stats.yeojohnson(i))
        return yeo_johnson_output
    
    @classmethod
    def drop_outliers(self, df, column):
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