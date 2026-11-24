import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Titanic 대시보드", layout="wide")

def filter_data(df, sexes, classes, age_range, ports):
    f = df[df["sex"].isin(sexes) & df["pclass"].isin(classes) & df["embarked"].isin(ports)]
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
    r = df.groupby("pclass")["survived"].mean().mul(100).round(1).reset_index()
    return px.bar(r, x="pclass", y="survived", text="survived", title="객실 등급별 생존율(%)")

def age_hist(df):
    return px.histogram(df, x="age", color="survived", barmode="overlay", opacity=0.6, title="생존 여부별 나이 분포")

LABELS = {"sex": "성별", "class": "객실 등급", "embarked": "승선 항구"}

def group_rate_bar(df, col):
    r = df.groupby(col)["survived"].mean().mul(100).round(1).reset_index()
    return px.bar(r, x=col, y="survived", text="survived", title=f"{LABELS[col]}별 생존율(%)")

def corr_heatmap(df):
    corr = df.select_dtypes(include="number").corr().round(2)
    return px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="수치형 열 상관 히트맵")

def iqr_outliers(df, col):
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((s < lo) | (s > hi)).sum())
    return count, lo, hi

@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

st.title("타이타닉, 누가 살아남았나?")

sexes = st.sidebar.multiselect("성별", ["male", "female"], ["male", "female"])
classes = st.sidebar.multiselect("객실 등급", [1, 2, 3], [1, 2, 3])
ages = st.sidebar.slider("나이", 0, 80, (0, 80))
ports = st.sidebar.multiselect("승선 항구", ["S", "C", "Q"], ["S", "C", "Q"])

if not sexes or not classes or not ports:
    st.warning("성별, 등급, 항구를 하나 이상 선택하세요.")
    st.stop()

f = filter_data(df, sexes, classes, ages, ports)
if len(f) == 0:
    st.warning("조건에 맞는 승객이 없습니다.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["개요", "관계", "이상치"])

with tab1:
    n, rate, age = kpi(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("승객 수", f"{n}명")
    c2.metric("생존율", f"{rate}%", f"{rate - 38.4:+.1f}%p")
    c3.metric("평균 나이", f"{age}세")
    left, right = st.columns(2)
    left.plotly_chart(class_rate_bar(f))
    right.plotly_chart(age_hist(f))
    col = st.selectbox("비교 기준", ["sex", "class", "embarked"], format_func=lambda c: LABELS[c])
    st.plotly_chart(group_rate_bar(f, col))
    with st.expander("데이터 보기"):
        st.dataframe(f)

with tab2:
    st.plotly_chart(corr_heatmap(f))
    st.caption("1에 가까우면 함께 커지고, -1에 가까우면 반대로 움직입니다.")

with tab3:
    ocol = st.selectbox("열 선택", ["age", "fare", "sibsp", "parch"])
    count, lo, hi = iqr_outliers(f, ocol)
    st.metric("IQR 이상치 수", f"{count}개")
    st.caption(f"정상 범위: {lo:.1f} ~ {hi:.1f}")
    st.plotly_chart(px.box(f, y=ocol, points="outliers", title=f"{ocol} 상자그림"))
