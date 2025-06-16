from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.future import select
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "postgresql+asyncpg://postgres:S1652321s@localhost:5432/postgres"
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    qualification = Column(Integer)
    specialization = Column(String)
    education = Column(Integer)
    age = Column(Integer)
    experience = Column(Float)
    productivity = Column(Float)

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    job_type = Column(String)
    normative_time  =Column(Float)
    required_qualification = Column(Integer)

class JobExperience(Base):
    __tablename__ = "jobs_experience"
    id = Column(Integer, primary_key=True)
    phys_id = Column(Integer)
    job_type_id = Column(Integer)
    field_experience = Column(Float)


class PredictionInput(BaseModel):
    data: dict
    time_limit: float

class EmployeeRead(BaseModel):
    id:int
    name:str
    qualification:int
    specialization:str
    education:int
    age:int
    experience:float
    productivity:float

    class Config:
        orm_mode = True

class JobRead(BaseModel):
    job_type:str
    normative_time:float
    required_qualification:int

    class Config:
        orm_mode = True

class JobExperienceRead(BaseModel):
    id:int
    phys_id:int
    job_type_id:int
    field_experience:float

    class Config:
        orm_mode = True


model_gb = None
features = None

@app.on_event("startup")
async def load_model():
    global model_gb, features
    loaded = joblib.load("ngb_model_features.pkl")
    model_gb = loaded["model"]
    features = loaded["features"]


@app.get("/employees", response_model=List[EmployeeRead])
async def get_employees():
    try:
        async with async_session() as session:
            result = await session.execute(select(Employee))
            employees = result.scalars().all()
            return employees
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Ошибка при получении сотрудников: {str(e)}")

@app.get("/jobs", response_model=List[JobRead])
async def get_jobs():
    try:
        async with async_session() as session:
            result = await session.execute(select(Job))
            jobs = result.scalars().all()
            return jobs
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Ошибка при получении работ: {str(e)}")

@app.get("/jobs_experience", response_model=List[JobExperienceRead])
async def get_job_experience():
    try:
        async with async_session() as session:
            result = await session.execute(select(JobExperience))
            jobs_exp = result.scalars().all()
            return jobs_exp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении опыта: {str(e)}")

@app.post("/predict")
async def predict(input: PredictionInput):
    try:
        df_pred = pd.DataFrame([input.data])
        df_pred = df_pred.reindex(columns=features, fill_value=0)
        pred_time = model_gb.predict(df_pred)[0]
        a = pred_time * 0.8
        b = pred_time
        c = pred_time * 1.2
        T = input.time_limit

        if T <= a:
            probability = 0.0
        elif T <= b:
            probability = ((T - a) ** 2) / ((b - a) * (c - a))
        elif T <= c:
            probability = 1 - ((c - T) ** 2) / ((c - b) * (c - a))
        else:
            probability = 1.0

        return {
            "prediction": round(pred_time, 3),
            "probability": round(probability * 100, 1),
            "a": round(a, 3),
            "b": round(b, 3),
            "c": round(c, 3)
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")
