import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import triang

API_URL = "http://localhost:8000"

st.set_page_config(layout="wide")

# --- Получение данных из FastAPI ---
@st.cache_data
def fetch_employees():
    return requests.get(f"{API_URL}/employees").json()

@st.cache_data
def fetch_jobs():
    return requests.get(f"{API_URL}/jobs").json()

@st.cache_data
def fetch_experience():
    return requests.get(f"{API_URL}/jobs_experience").json()


def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj
    
df_exec = pd.DataFrame(fetch_employees())
df_jobs = pd.DataFrame(fetch_jobs())
df_jobs_exp = pd.DataFrame(fetch_experience())

# --- UI ---
st.sidebar.title("Выберите исполнителя")
name = st.sidebar.selectbox("", df_exec['name'])
isp = df_exec[df_exec['name'] == name].iloc[0]

st.sidebar.title("Выберите тип работы")
job_type = st.sidebar.selectbox("", df_jobs['job_type'])
job = df_jobs[df_jobs['job_type'] == job_type].iloc[0]

job_exp_row = df_jobs_exp[(df_jobs_exp['job_type_id'] == job_type) & (df_jobs_exp['phys_id'] == isp['id'])]
field_experience = int(job_exp_row.iloc[0]['experience']) if not job_exp_row.empty else 0

# --- Ввод параметров ---
sost = st.sidebar.slider("Состояние объекта", 1, 5, 4)
plan = st.sidebar.radio("Плановая / внеплановая", [0, 1], format_func=lambda x: "Плановая" if x == 0 else "Внеплановая")
zapas = st.sidebar.selectbox("Наличие запчастей", ["Есть", "Нет"])
workload = st.sidebar.slider("Предварительная нагрузка (задач до)", 0, 4, 1)
complexity = st.sidebar.slider("Сложность", 1, 5, int(job.get('complexity', 3)))
normative_time = float(job['normative_time'])
required_qualification = int(job['required_qualification'])

season = st.sidebar.selectbox("Время года", ["весна", "лето", "осень", "зима"])
place = st.sidebar.selectbox("Место выполнения работы", ["депо", "путь", "станция"])
shift = st.sidebar.selectbox("Смена", ["день", "ночь"])

time_limit = st.sidebar.number_input("Время выполнения (часы)", 1.0, 48.0, normative_time, step=0.5)

if st.sidebar.button("Рассчитать"):
    # --- Подготовка данных ---
    data = {
        'Возраст': isp['age'],
        'Стаж': isp['experience'],
        'Стаж в области': field_experience,
        'Коэффициент производительности': isp['productivity'],
        'Квалификация': isp['qualification'],
        'Образование': isp['education'],
        'Базовый норматив (часы)': normative_time,
        'Сложность': complexity,
        'Плановая (0) / Внеплановая (1)': plan,
        'Состояние объекта': sost,
        'Наличие запчастей (1-да,0-нет)': int(zapas == "Есть"),
        'Предварительная нагрузка (задач до)': workload,
        'Требуемый разряд': required_qualification,
        'Специализация_Механик': int(isp['specialization'] == "Механик"),
        'Тип работы_ТО-2': int(job_type == "ТО-2"),
        'Время года_лето': int(season == "лето"),
        'Место выполнения работы_путь': int(place == "путь"),
        'Смена_Ночь': int(shift == "ночь"),
    }
    data = to_python_types(data)
    response = requests.post(f"{API_URL}/predict", json={"data": data, "time_limit": time_limit}).json()
    print(data)
    pred_time = response["prediction"]
    prob = response["probability"]
    a, b, c = response["a"], response["b"], response["c"]
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<div style='font-size:2.5em;'>С вероятностью <b>{prob}%</b> задача будет выполнена за <b>{time_limit} часов</b>.</div>", unsafe_allow_html=True)

    rv = triang((b - a)/(c - a), loc=a, scale=c - a)
    x = np.linspace(a, c, 500)
    y = rv.pdf(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Плотность вероятности", color="dimgray")
    ax.fill_between(x, y, alpha=0.3, color="lightgray")
    ax.axvline(x=time_limit, color="red", linestyle="--", label=f"Порог: {time_limit} ч")
    ax.set_title("Треугольное распределение времени выполнения")
    ax.set_xlabel("Время")
    ax.set_ylabel("Плотность")
    ax.grid(True)
    ax.legend()
    col1, col2, col3 = st.columns([1, 2, 1])  # Центральная колонка в 2 раза шире
    with col2:
        st.pyplot(fig, use_container_width=False)