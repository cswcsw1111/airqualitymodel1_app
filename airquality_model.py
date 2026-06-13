import streamlit as st
import pandas as pd
import numpy as np
import joblib  # .pkl 모델 파일을 불러오기 위한 핵심 라이브러리
import os

# =================================================================
# 1. 세련된 웹 대시보드 레이아웃 및 디자인 설정
# =================================================================
st.set_page_config(
    page_title="EcoInformatics 대기질 예측 AI 대시보드",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세련된 UI를 위한 커스텀 CSS 스타일 정의
st.markdown("""
    <style>
    .main-title { font-size: 2.4rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    .category-title { font-size: 1.3rem; font-weight: 700; color: #2563EB; margin-top: 1rem; margin-bottom: 1rem; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.3rem;}
    .result-card { padding: 1.5rem; border-radius: 0.75rem; text-align: center; margin-bottom: 1.5rem; color: white; font-weight: bold; font-size: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# 2. 저장된 .pkl 파일로부터 랜덤포레스트 모델 로드
# =================================================================
model_filename = "airquality_model.pkl"

if os.path.exists(model_filename):
    # 폴더에 있는 airquality_model.pkl 파일을 읽어옵니다.
    rf_model = joblib.load(model_filename)
else:
    st.error(f"❌ '{model_filename}' 파일을 찾을 수 없습니다. 같은 폴더에 pkl 파일이 있는지 확인해 주세요.")
    st.stop()

# =================================================================
# 3. 사이드바 프로필 및 대시보드 타이틀 구성
# =================================================================
st.sidebar.image("https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=500", use_container_width=True)
st.sidebar.markdown("### 🔬 환경 화학 및 데이터 분석")
st.sidebar.info(
    "본 대시보드는 대기 중 화학 가스 농도 및 물리적 기상 요인, "
    "지리적 변수를 다차원적으로 분석하여 실시간 대기질 등급을 분류하는 AI 시스템입니다."
)

st.markdown('<div class="main-title">🌤️ 대기 정보학(Eco-Informatics) 기반 대기질 분류 AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">학습이 완료된 랜덤포레스트 모델(airquality_model.pkl)을 호출하여 실시간으로 대기질을 예측합니다.</div>', unsafe_allow_html=True)

# =================================================================
# 4. 레이아웃 분할 및 슬라이더 입력창 배치
# =================================================================
col_input, col_result = st.columns([5, 4], gap="large")

with col_input:
    st.markdown('<div class="category-title">🌡️ 1. 기상 및 인구학적 환경 요인</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider("기온 (°C)", min_value=-10.0, max_value=45.0, value=18.0, step=0.1)
    with c2:
        humid = st.slider("습도 (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
    with c3:
        pop_dens = st.slider("인구밀도 (명/km²)", min_value=10.0, max_value=1500.0, value=120.0, step=10.0)
        
    st.markdown('<div class="category-title">🏭 2. 미세먼지 및 지리적 위험도</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        pm25 = st.slider("초미세먼지 PM₂.₅ (µg/m³)", min_value=0.0, max_value=250.0, value=12.0, step=1.0)
    with c5:
        pm10 = st.slider("미세먼지 PM₁₀ (µg/m³)", min_value=0.0, max_value=400.0, value=25.0, step=1.0)
    with c6:
        ind_prox = st.slider("산업단지 근접도 (km)", min_value=0.0, max_value=30.0, value=15.0, step=0.1)

    st.markdown('<div class="category-title">🧪 3. 대기 중 가스성 화학 물질 농도</div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        no2 = st.slider("이산화질소 NO₂ (ppb)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
    with c8:
        so2 = st.slider("이산화황 SO₂ (ppb)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
    with c9:
        co = st.slider("일산화탄소 CO (ppm)", min_value=0.0, max_value=15.0, value=0.5, step=0.1)

# 모델이 학습했을 때와 정확히 동일한 컬럼 명칭 및 순서로 데이터프레임 생성
input_data = pd.DataFrame(
    [[temp, humid, pm25, pm10, no2, so2, co, ind_prox, pop_dens]],
    columns=['기온', '습도', '초미세먼지', '미세먼지', '이산화질소', '이산화황', '일산화탄소', '산업단지_근접도', '인구밀도']
)

# =================================================================
# 5. 실시간 AI 모델 예측 및 시각적 결과 출력
# =================================================================
with col_result:
    st.markdown('<div class="category-title">📊 4. AI 실시간 진단 및 분석 결과</div>', unsafe_allow_html=True)
    
    # 불러온 pkl 모델로 예측 수행
    predicted = rf_model.predict(input_data)
    prob = rf_model.predict_proba(input_data)
    
    # 등급별 UI 테마 매핑 설정 (색상 및 이모지)
    status_config = {
        0: {"label": "Good (좋음) 🟢", "color": "#10B981"},      # Emerald Green
        1: {"label": "Moderate (보통) 🟡", "color": "#F59E0B"},  # Amber Yellow
        2: {"label": "Poor (나쁨) 🟠", "color": "#EF4444"},      # Light Red
        3: {"label": "Hazardous (위험) 🔴", "color": "#7F1D1D"}  # Dark Red
    }
    
    current_status = status_config[predicted[0]]
    
    # 결과 알림 카드 출력 (예측 등급에 따라 배경 색상이 동적으로 바뀜)
    st.markdown(
        f'<div class="result-card" style="background-color: {current_status["color"]};">'
        f'종합 대기질 상태: {current_status["label"]}'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    # 확률 분포 시각화 (프로그레스 바를 통해 세련되게 표현)
    st.write("**💡 각 등급별 분류 확률 산출 지표**")
    
    labels = ['Good (좋음)', 'Moderate (보통)', 'Poor (나쁨)', 'Hazardous (위험)']
    
    for i, label in enumerate(labels):
        prob_val = prob[0][i]
        st.caption(f"{label} : {prob_val*100:.1f}%")
        st.progress(float(prob_val))

    st.divider()
    st.markdown(
        "🔎 **화학적 인자 분석 팁:** 가스성 성분($NO_2, SO_2, CO$)과 미세먼지 수치 슬라이더를 우측으로 "
        "조정할수록 모델 내부 엔트로피 분기가 작동하여 **Poor** 및 **Hazardous** 등급 확률이 우상향하게 됩니다."
    )
