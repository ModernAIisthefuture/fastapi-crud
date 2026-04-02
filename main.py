from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from prometheus_fastapi_instrumentator import Instrumentator

# DB
engine = create_engine("mysql+pymysql://root:root@localhost/loan_app")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# FastAPI
app = FastAPI()
#Instrumentator().instrument(app).expose(app)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Table
class Loan(Base):
    __tablename__ = "loans"
    id = Column(String(10), primary_key=True)
    gender = Column(String(10))
    married = Column(String(10))
    dependents = Column(String(10))
    education = Column(String(20))
    self_employed = Column(String(10))
    applicant_income = Column(Float)
    coapplicant_income = Column(Float)
    loan_amount = Column(Float)
    loan_amount_term = Column(Float)
    credit_history = Column(Float)
    property_area = Column(String(20))
    loan_status = Column(String(5))

Base.metadata.create_all(bind=engine)

# ML
model = None
encoders = {}
cat_cols = ['gender','married','dependents','education','self_employed','property_area']

def preprocess(df, fit=False):
    global encoders
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col])
    return df

# CRUD
@app.post("/loans/")
def create(data: dict):
    db = SessionLocal()
    loan = Loan(**data)
    db.add(loan)
    db.commit()
    db.refresh(loan)
    db.close()
    return loan

@app.get("/loans/{id}")
def read(id: str):
    db = SessionLocal()
    loan = db.query(Loan).get(id)
    db.close()
    if not loan:
        raise HTTPException(404)
    return loan

@app.delete("/loans/{id}")
def delete(id: str):
    db = SessionLocal()
    loan = db.query(Loan).get(id)
    db.delete(loan)
    db.commit()
    db.close()
    return {"deleted": id}

@app.put("/loans/{id}")
def update(id: str, data: dict):
    db = SessionLocal()

    loan = db.query(Loan).filter(Loan.id == id).first()

    if not loan:
        db.close()
        raise HTTPException(status_code=404, detail="Loan not found")

    for key, value in data.items():
        setattr(loan, key, value)

    db.commit()
    db.refresh(loan)
    db.close()

    return loan

# Predict
@app.post("/predict/")
def predict(data: dict):
    global model
    df = pd.DataFrame([data])
    df = preprocess(df)
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}

# Retrain
@app.post("/retrain/")
def retrain():
    global model
    db = SessionLocal()
    df = pd.read_sql("SELECT * FROM loans", engine)
    db.close()

    df['loan_status'] = df['loan_status'].map({'Y':1,'N':0})
    X = df.drop(['id','loan_status'], axis=1)
    y = df['loan_status']

    X = preprocess(X, fit=True)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    acc = model.score(X_test,y_test)
    joblib.dump(model,"model.pkl")

    return {"accuracy": acc}