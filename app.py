# ===== 설정 =====
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Titanic 대시보드", layout="wide")

# ===== 함수 =====
def filter_data(df, sexes, classes, age_range, ports):
    f = df[df["sex"].isin(sexes)
           & df["pclass"].isin(classes)
           & df["embarked"].isin(ports)]
    if age_range != (0, 80):
        f = f[f["age"].between(*age_range)]
    return f

def kpi(df):
    if len(df) == 0:
        return 0, 0, 0
    n = len(df)
    rate = round(df["survived"].mean() * 100, 1)
    age = round(df["age"].mean(), 1)
    return n, rate, age

def class_rate_bar(df):
    r = (df.groupby("pclass")["survived"]
           .mean().mul(100).round(1)
           .reset_index())
    fig = px.bar(r, x="pclass", y="survived", text="survived",
                 title="객실 등급별 생존율(%)")
    return fig

def age_hist(df):
    fig = px.histogram(df, x="age", color="survived", barmode="overlay",
                       opacity=0.6, title="생존 여부별 나이 분포")
    return fig

LABELS = {"sex": "성별", "class": "객실 등급", "embarked": "승선 항구"}

def group_rate_bar(df, col):
    r = (df.groupby(col)["survived"]
           .mean().mul(100).round(1)
           .reset_index())
    fig = px.bar(r, x=col, y="survived", text="survived",
                 title=f"{LABELS[col]}별 생존율(%)")
    return fig

def corr_heatmap(df):
    corr = df.select_dtypes(include="number").corr().round(2)
    fig = px.imshow(corr,
