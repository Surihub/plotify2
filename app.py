import streamlit as st
import pandas as pd
import seaborn as sns
import utils as eda  # eda 모듈 임포트
import datetime
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

st.header("🌲Wep app for EDA")
st.success("🎈EDA(Exploratory Data Analysis, 탐색적 데이터 분석)이란 간단한 그래프로 데이터의 특징과 패턴을 찾아내어 데이터를 탐구하기 위한 과정입니다. 왼쪽의 사이드바에서 데이터를 선택하거나 업로드하고, 순서에 따라 탐색을 진행해보세요. **단, 입력하는 데이터는 원자료(raw data)의 형태**여야 합니다. \n\n✉ 버그 및 제안사항 등 문의: sbhath17@gmail.com(황수빈), code: [github](https://github.com/Surihub/plot)")

# 스트림릿 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'selected_columns' not in st.session_state:
    st.session_state['selected_columns'] = None
if 'user_column_types' not in st.session_state:
    st.session_state['user_column_types'] = None
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'columns_selected' not in st.session_state:
    st.session_state['columns_selected'] = False
if 'types_set' not in st.session_state:
    st.session_state['types_set'] = False
if 'transformations' not in st.session_state:
    st.session_state['transformations'] = {}
if 'viz' not in st.session_state:
    st.session_state['viz'] = {} 

dataset_name = st.sidebar.selectbox("분석하고 싶은 데이터를 선택해주세요!",
    sns.get_dataset_names(), index = 16, help = "처음이시라면, 귀여운 펭귄들의 데이터인 'penguins'를 추천드려요😀")
with st.sidebar:
    uploaded_file = st.file_uploader("혹은, 파일을 업로드해주세요!", type=["csv"], help = 'csv파일만 업로드됩니다😥')
with st.sidebar:
    if uploaded_file is not None:
        mydata = "업로드한 데이터"
    else:
        mydata = dataset_name
    if st.checkbox(f'**{mydata}** 불러오기'):
        # df = sns.load_dataset(dataset_name)
        df = eda.load_data(dataset_name, uploaded_file)
       
st.subheader("👀 데이터 확인하기")
# st.write(df)
try:
    if df is not None:
        st.session_state['df'] = df
        st.session_state['data_loaded'] = True
        st.write("데이터 로드 완료! 불러온 데이터셋은 다음과 같습니다. ")
        st.write(df.head())
        with st.expander('전체 데이터 보기'):
            st.write(df)
except:
    st.error("사이드바에서 먼저 데이터를 선택 후 <데이터 불러오기> 버튼을 클릭해주세요. ")
# st.write(st.session_state['data_loaded'])
# 2. 열 선택
if st.session_state['data_loaded']:
    df = st.session_state['df']
    st.subheader("👈 분석할 열 선택하기")
    st.success(f"이 데이터는 {df.shape[0]}개의 행(가로줄), {df.shape[1]}개의 열(세로줄)로 이뤄진 데이터네요! 그럼, 위의 데이터셋에서, 분석할 열만 선택해주세요.")


from stemgraphic import stem_graphic
# 3. 데이터 시각화
if st.session_state['data_loaded']:
    tab1, tab2 = st.tabs(["한 개의 시각화", "두 개의 변량 시각화"])

    with tab1:
        st.subheader("📈 한 변량 데이터 시각화")
        st.success("위에서 나타낸 패턴을 바탕으로, 한 열만을 골라 다양하게 시각화해보면서 추가적으로 탐색해봅시다. ")
        colu1, colu2 = st.columns(2)
        with colu1:
            selected_columns = st.radio('분석하고자 하는 열을 선택하세요:', st.session_state['df'].columns.tolist())
        with colu2:
            graph_type = st.radio("그래프 종류를 선택해주세요. ", ["막대그래프", "원그래프", "띠그래프", "꺾은선그래프", "줄기와잎그림", "히스토그램", "상자그림"])
        

        st.session_state['selected_columns'] = selected_columns
        if st.button('열 선택 완료!'):
            st.session_state['columns_selected'] = True
            st.success("열 선택 완료!")
            
        df1 = df[st.session_state['selected_columns']]

        if graph_type =="히스토그램":
            if pd.api.types.is_float_dtype(df1):
                wid = (df1.max()-df1.min())/10
            else:
                wid = 100
            binwidth = st.number_input("계급의 크기를 입력해주세요.", value = wid)
        else:
            binwidth = None
        st.session_state['df1'] = df1

        st.write(graph_type+"를 그린 결과입니다. 저장하려면 버튼을 클릭하세요.")
        fig = eda.선택해서_그래프_그리기(pd.DataFrame(df1), graph_type, binwidth)

        # 그림으로 저장
        st.session_state['graph_type'] = graph_type
        st.session_state['fig'] = fig
        fig_path = "fig.png"
        st.session_state.fig.savefig(fig_path)

        with open("fig.png", "rb") as file:
            btn = st.download_button(
                    label="그래프 다운로드 받기[일변량]",
                    data=file,
                    file_name=f"{selected_columns}_{graph_type}.png",
                    mime="image/png")
        st.session_state['viz'] = True
        # 띠그래프 비율 표시 추가
        # 평균 추가할지?

    with tab2:
        st.subheader("📈 두 개의 변량 데이터 시각화")
        st.success("위에서 나타낸 패턴을 바탕으로, 가로축, 세로축을 선택하여 다양하게 시각화해보면서 추가적으로 탐색해봅시다. ")
        x_var_col, y_var_col = st.columns(2)
        with x_var_col:
            x_var = st.radio('가로축 변수를 선택하세요:', st.session_state['df'].columns.tolist())
        with y_var_col:
            y_var = st.radio('세로축 변수를 선택하세요:', st.session_state['df'].columns.tolist())
        if x_var==y_var:
            st.error("서로 다른 변수를 선택해주세요.")
        else:
            graph_type_2 = st.radio("이변량그래프 종류를 선택해주세요.", ["막대그래프", "꺾은선그래프", "히스토그램", "상자그림", "산점도"])

            st.write(graph_type_2+"를 그린 결과입니다. 저장하려면 버튼을 클릭하세요.")
            fig = eda.선택해서_그래프_그리기_이변량(df, x_var, y_var, graph_type_2, binwidth)

            # 그림으로 저장
            st.session_state['graph_type_2'] = graph_type_2
            st.session_state['fig'] = fig
            fig_path = "fig.png"
            st.session_state.fig.savefig(fig_path)

            with open("fig.png", "rb") as file:
                btn = st.download_button(
                        label="그래프 다운로드 받기[이변량]",
                        data=file,
                        file_name=f"{selected_columns}_{graph_type_2}.png",
                        mime="image/png")
            st.session_state['viz'] = True

