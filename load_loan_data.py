import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://root:root@localhost/loan_app")

df = pd.read_csv("loan_data_set.csv")

df.fillna({
    'Gender':'Male','Married':'Yes','Dependents':'0','Self_Employed':'No',
    'LoanAmount': df['LoanAmount'].median(),
    'Loan_Amount_Term':360,
    'Credit_History':1.0
}, inplace=True)

df.rename(columns={
    'Loan_ID': 'id',
    'Gender':'gender',
    'Married':'married',
    'Dependents':'dependents',
    'Education':'education',
    'Self_Employed':'self_employed',
    'ApplicantIncome':'applicant_income',
    'CoapplicantIncome':'coapplicant_income',
    'LoanAmount':'loan_amount',
    'Loan_Amount_Term':'loan_amount_term',
    'Credit_History':'credit_history',
    'Property_Area':'property_area',
    'Loan_Status':'loan_status'
}, inplace=True)

with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS loans (
        id VARCHAR(10),
        gender VARCHAR(10),
        married VARCHAR(10),
        dependents VARCHAR(10),
        education VARCHAR(20),
        self_employed VARCHAR(10),
        applicant_income FLOAT,
        coapplicant_income FLOAT,
        loan_amount FLOAT,
        loan_amount_term FLOAT,
        credit_history FLOAT,
        property_area VARCHAR(20),
        loan_status VARCHAR(5)
    );
    """))

df.to_sql("loans", con=engine, if_exists="append", index=False)

print("Data inserted successfully!")