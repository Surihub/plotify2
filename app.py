import streamlit as st
import pandas as pd
import seaborn as sns
import utils as eda  # eda 모듈 임포트
import datetime
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import deepl
from stemgraphic import stem_graphic

# CSS를 사용하여 Streamlit 앱의 왼쪽 및 오른쪽 패딩 제거
css_style = """
    <style>
        .css-18e3th9 {
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        .stApp {
            padding-top: 0px;
            padding-bottom: 0px;
            padding-left: 0px;
            padding-right: 0px;
        }
    </style>
"""
st.markdown(css_style, unsafe_allow_html=True)

st.header("📌 데이터 과학을 위한 공학도구", help="🎈EDA(Exploratory Data Analysis, 탐색적 데이터 분석)이란 간단한 그래프로 데이터의 특징과 패턴을 찾아내어 데이터를 탐구하기 위한 과정입니다. 왼쪽의 사이드바에서 데이터를 선택하거나 업로드하고, 순서에 따라 탐색을 진행해보세요. **단, 입력하는 데이터는 원자료(raw data)의 형태**여야 합니다. \n\n✉ 버그 및 제안사항 등 문의: sbhath17@gmail.com(황수빈), code: [github](https://github.com/Surihub/plot)")
with st.chat_message(name = "human", avatar="🧑‍💻"):
    st.write("탐색적 데이터 분석을 위한 공학도구에 오신 것을 환영합니다. 왼쪽 사이드바에서 자료를 불러와주세요.")

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

st.sidebar.write("# 🎁 데이터 선택하기")
dataset_name = st.sidebar.selectbox("분석하고 싶은 데이터를 선택해주세요!",
    # sns.get_dataset_names(),
    # index = 16, 
    ['penguins_kor', 'tips_kor', 'healthcare_kor', 'world_happiness_report_2021'],
    help = "처음이시라면, 귀여운 펭귄들의 데이터인 'penguins_kor'를 추천드려요😀")

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
        if st.checkbox(f'**{mydata}** 조금만 불러오기'):
            n_sample = st.number_input(f"이 데이터는 총 {df.shape[0]}행이네요. 임의로 추출할 표본 개수를 입력해주세요.", value = 30, step=1)
            if df.shape[0]>n_sample:
                df = df.sample(n=n_sample, random_state=42)
       
st.subheader("👀 데이터 확인하기")
# st.write(df)
try:
    if df is not None:
        st.session_state['df'] = df
        st.session_state['data_loaded'] = True
        st.success('데이터 로드 완료!👍🏻 불러온 데이터셋은 다음과 같습니다.')
        if dataset_name=="penguins":
            with st.expander("펭귄 데이터셋ㅁㄴㅇㄹ에 대한asdf 설명을 보려면 여기를 클릭하세요."):
                st.write("Wkwis!")
        st.write(df.head())
        
except:
    st.error("사이드바에서 먼저 데이터를 선택 후 <데이터 불러오기> 버튼을 클릭해주세요. ")
# st.write(st.session_state['data_loaded'])
# 2. 열 선택
if st.session_state['data_loaded']:
    df = st.session_state['df']
    # st.subheader("👈 분석할 열 선택하기")
    st.info(f"이 데이터는 {df.shape[0]}개의 행(가로줄), {df.shape[1]}개의 열(세로줄)로 이뤄진 데이터네요! 전체 데이터는 아래를 눌러 확인해보세요. 그래프를 그리기 위해서는 그 아래의 버튼을 클릭해주세요. ")
    with st.expander('전체 데이터 보기'):
        st.write(df)

    # 데이터 시각화를 위한 '다음 버튼' 생성
    if st.button('시각화를 통해 데이터 탐색하기', type = 'primary', use_container_width = True):
        # 버튼이 클릭되면 'show_visualization' 상태를 True로 설정
        st.session_state['show_visualization'] = True

# 3. 데이터 시각화
if st.session_state.get('show_visualization', False):
    tab1, tab2 = st.tabs(["한 개의 시각화", "두 개의 변량 시각화"])

    with tab1:
        st.subheader("📈 한 변량 데이터 시각화")
        st.success("위에서 나타낸 패턴을 바탕으로, 한 열만을 골라 다양하게 시각화해보면서 추가적으로 탐색해봅시다. ")
        colu1, colu2 = st.columns(2)
        with colu1:
            selected_columns = st.selectbox('분석하고자 하는 열을 선택하세요:', st.session_state['df'].columns.tolist())
        with colu2:
            graph_type = st.selectbox("그래프 종류를 선택해주세요. ", ["막대그래프", "원그래프", "띠그래프", "꺾은선그래프", "줄기와잎그림", "히스토그램", "도수분포다각형", "상자그림"])
        

        st.session_state['selected_columns'] = selected_columns
        # if st.button('열 선택 완료!'):
        #     st.session_state['columns_selected'] = True
        #     st.success("열 선택 완료!")        

        # 그래프 옵션 ########
        st.write("----")
        graph_option_1, graph_1 = st.columns([1/3, 2/3])
        with graph_option_1:
            option = None  # 옵션 초기화
            st.subheader("⚙️그래프 옵션")
            graph_title_1 = st.text_input("그래프 제목을 입력해주세요.")
            
        with graph_1:
            st.subheader("📊그래프 보기")
        ########
        df1 = df[st.session_state['selected_columns']]

        # st.session_state['df1'] = df1
        ################################################################
        if graph_type == "막대그래프":
            with graph_option_1:
                with st.expander("범주 순서 지정하려면 클릭하세요"):
                    option = []
                    horizontal = st.checkbox("가로로 그리기")
                    option.append(horizontal)

                    ## 값 순서대로 클릭하기 일단 없애기
                    # order = st.multiselect("값들을 순서대로 클릭해주세요.", options = df1.unique(), default = df1.unique())
                    # option.append(order)

        elif graph_type in ["히스토그램", "도수분포다각형"]:
            if pd.api.types.is_float_dtype(df1):
                wid = (df1.max()-df1.min())/10
            else:
                wid = 100
            with graph_option_1:
                option = st.number_input("변량의 계급의 크기를 입력해주세요.", value = wid)


        with graph_option_1:
            rot_angle = st.number_input("가로축 글씨 회전시키기. ", min_value = 0, max_value = 90, step = 45)
        fig = eda.선택해서_그래프_그리기(pd.DataFrame(df1), graph_type, option = option, rot_angle = rot_angle)

        with graph_1:
            plt.title(graph_title_1, fontsize=15)
            plt.xticks(rotation = rot_angle)
            st.pyplot(fig)

        # 그림으로 저장
        st.session_state['graph_type'] = graph_type
        st.session_state['fig'] = fig
        with graph_option_1:
            img_type = st.radio("이미지 파일 형식 선택 ",['png', 'svg'])

        fig_path = f"fig.{img_type}"
        st.session_state.fig.savefig(fig_path)
        with graph_1:
            with open(fig_path, "rb") as file:
                btn = st.download_button(
                    label="그래프 다운로드 받기[일변량]",
                    data=file,
                    type = 'primary', 
                    use_container_width=True,
                    file_name=f"{selected_columns}_{graph_type}.{img_type}",
                    mime=f"image/{'svg+xml' if img_type == 'svg' else img_type}"
                )
        st.session_state['viz'] = True
        # 띠그래프 비율 표시 추가
        # 평균 추가할지?
        st.write("-----")
        st.subheader("🖋️ 데이터 요약하기")
        summary, table = st.columns(2)

        # 선택된 데이터 열을 이용하기
        with summary:
            if pd.api.types.is_numeric_dtype(df1):
                summ = eda.summarize(df1)
                st.write(summ)
            else:
                summ_cat = eda.table_cat(df1)
                st.write(summ_cat)
        # 요약 표 표시
        with table:
            if pd.api.types.is_numeric_dtype(df1):
                if graph_type in ["히스토그램", "도수분포다각형"]:
                    st.write(eda.table_num(df1, option))
                else:
                    st.error("그래프 종류를 히스토그램 혹은 도수분포다각형으로 변경해주세요.")
            else:
                st.write(" ")




    with tab2:
        st.subheader("📈 두 개의 변량 데이터 시각화")
        st.success("위에서 나타낸 패턴을 바탕으로, 가로축, 세로축을 선택하여 다양하게 시각화해보면서 추가적으로 탐색해봅시다. ")
        # try: # 맨 나중에 처리
        x_var_col, y_var_col, select_graph = st.columns(3)
        col_list = st.session_state['df'].columns.tolist()
        with x_var_col:
            x_var = st.selectbox('가로축 변수를 선택하세요:', col_list)
        with y_var_col:
            y_var = st.selectbox('세로축 변수를 선택하세요(그룹):', col_list)
        
        st.session_state['x_var'] = x_var
        st.session_state['y_var'] = y_var

        if x_var and y_var and x_var == y_var:
            st.error("서로 다른 변수를 선택해주세요.")
        elif x_var and y_var:
            df = st.session_state['df']
            with select_graph:
                graph_type_2 = st.selectbox("이변량그래프 종류를 선택해주세요.", ["막대그래프", "꺾은선그래프", "히스토그램", "도수분포다각형", "상자그림", "산점도"])

            # 그래프 옵션 ########
            st.write("----")
            graph_option, graph = st.columns([1/3, 2/3])
            with graph_option:
                st.subheader("⚙️그래프 옵션")
                graph_title_2 = st.text_input("그래프 제목을 입력해주세요.    ")
            with graph:
                st.subheader("📊그래프 보기")
            ########
            if graph_type_2 != None:
                if graph_type_2 == "산점도":
                    option = []
                    with graph_option:
                        scatter_group_color = st.checkbox("색으로 구분하기")# 범주
                        if scatter_group_color:
                            option_1 = st.selectbox("색으로 구분할 옵션을 선택해주세요.",df.columns.tolist())
                        else:
                            option_1 = None

                        scatter_group_shape = st.checkbox("모양으로 구분하기")# 범주
                        if scatter_group_shape:
                            option_2 = st.selectbox("모양으로 구분할 옵션을 선택해주세요.",df.columns.tolist())
                        else:
                            option_2 = None                            

                        scatter_group_size = st.checkbox("크기로 구분하기")# 수치
                        if scatter_group_size:
                            option_3 = st.selectbox("크기 기준을 선택해주세요.",df.columns.tolist())
                        else:
                            option_3 = None

                        trend_line_button = st.checkbox("추세선 보이기")
                        # hue 구분 옵션

                        if trend_line_button:
                            option_4 = True
                        else:
                            option_4 = False
                        option.append(option_1)
                        option.append(option_2)
                        option.append(option_3)
                        option.append(option_4)

                elif graph_type_2 =="꺾은선그래프":
                    # 세로축 범위 옵션
                    with graph_option:
                        if st.checkbox("0부터 표시합니다."):
                            option = None
                        else:
                            option = True

                elif graph_type_2 in ["히스토그램", "도수분포다각형"]:
                    if pd.api.types.is_numeric_dtype(df[x_var]):
                        wid = (df[x_var].max()-df[x_var].min())/10
                    else:
                        wid = 100
                    with graph_option:
                        option = st.number_input("공통된 계급의 크기를 입력해주세요.", value = wid)

                elif graph_type_2 == "막대그래프":
                    with graph_option:
                        option = st.checkbox("누적막대그래프")

                else:
                    option = None

                with graph_option:
                    rot_angle = st.number_input("가로축 글씨 회전시키기", min_value = 0, max_value = 90, step = 45)
                fig = eda.선택해서_그래프_그리기_이변량(df, x_var, y_var, graph_type_2, option=option, rot_angle = rot_angle)
                with graph:
                    plt.title(graph_title_2)
                    plt.xticks(rotation = rot_angle)
                    st.pyplot(fig)
                # 그림으로 저장
                st.session_state['graph_type_2'] = graph_type_2
                st.session_state['fig'] = fig

                with graph_option:
                    img_type = st.radio("이미지 파일 형식 선택",['png', 'svg'], help="그래프의 색, 글씨 크기, 범례 위치 등 세부적인 요소를 수정하려면 svg로 다운받아보세요. 다운받은 이미지를 파워포인트에서 열면 모두 그룹 해제하여 하나하나 수정할 수 있어요.")

                fig_path = f"fig.{img_type}"
                st.session_state.fig.savefig(fig_path)


                with graph:
                    with open(fig_path, "rb") as file:
                        btn = st.download_button(
                            label="그래프 다운로드 받기[이변량]",
                            data=file,
                            type = 'primary', 
                            use_container_width=True,
                            file_name=f"{x_var}_{y_var}_{graph_type_2}.{img_type}",
                            mime=f"image/{'svg+xml' if img_type == 'svg' else img_type}"
                        )
                    
                
                st.session_state['viz'] = True

        # except Exception as e:
        #     translator = deepl.Translator(st.secrets['deepl']['key'])
        #     error_message = translator.translate_text(f"{e}", target_lang="KO")
        #     st.error(f"그래프를 그릴 수 없습니다.  \n오류메시지{e}\n\n오류메시지(kor){error_message}")

        st.write("-----")
        x_var = st.session_state['x_var']
        y_var = st.session_state['y_var']
        st.subheader("🖋️ 데이터 요약하기")
  

        x_is_numeric = pd.api.types.is_numeric_dtype(df[x_var])
        y_is_numeric = pd.api.types.is_numeric_dtype(df[y_var])

        # 수치형 여부에 따라 메시지 출력
        if x_is_numeric and y_is_numeric:
            # 피어슨 상관계수 계산
            correlation = df[[x_var, y_var]].corr(method='pearson').iloc[0, 1]
            st.write(f"피어슨 상관계수 ({x_var} & {y_var}): {correlation:.3f}")
            
        elif not x_is_numeric and y_is_numeric:
            st.write(f"{x_var} & {y_var}의 통계량")
            summary_stats = df.groupby(x_var)[y_var].agg(['mean', 'median', 'std']).reset_index()
            summary_stats.columns = [x_var]+["평균", "중앙값", '표준편차']
            st.write(summary_stats)
        elif x_is_numeric and not y_is_numeric:
            st.write(f"{y_var} & {x_var}의 통계량")
            summary_stats = df.groupby(y_var)[x_var].agg(['mean', 'median', 'std']).reset_index()
            summary_stats.columns = [y_var]+["평균", "중앙값", '표준편차']
            st.write(summary_stats)
        elif not x_is_numeric and not y_is_numeric:
            st.write(f"{x_var} & {y_var}")
            st.write("빈도표")
            st.write(pd.crosstab(index=df[x_var], columns=df[y_var], margins=True, margins_name="Total"))

