import pandas as pd
import yaml
from sqlalchemy import create_engine

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
    
if __name__ == "__main__":
    creds = get_credentials('credentials.yaml')
    RDSDBConn = RDSDatabaseConnector(creds)
    engine = RDSDBConn.create_conn()
    df = RDSDBConn.get_dataframe(engine)
    RDSDBConn.save_to_file(df)