import streamlit as st
import pickle
import numpy as np

# 1️⃣ 모델 불러오기 (캐싱 기능을 써서 속도 최적화)
@st.cache_resource
def load_model():
    with open("titanic_model.pkl", "rb") as f:
        return pickle.load(f)

model, label_encoder = load_model()

# 2️⃣ 웹페이지 제목 및 설명
st.title("🚢 타이타닉 생존 예측 앱")
st.write("성별과 나이를 입력하면 생존 확률을 알려줍니다.")

# 3️⃣ 사용자 입력 받기 (HTML Form 대신 파이썬 함수 사용)
sex = st.selectbox("성별을 선택하세요", ["male", "female"])
age = st.number_input("나이를 입력하세요", min_value=0.0, max_value=100.0, value=20.0)

# 4️⃣ 버튼 클릭 시 예측 실행
if st.button("결과 확인하기"):
    # 데이터 전처리
    sex_encoded = label_encoder.transform([sex])[0]
    X_input = np.array([[sex_encoded, age]])
    
    # 예측 수행
    survival_prob = model.predict_proba(X_input)[0][1]
    
    # 결과 화면 출력
    st.success(f"이 승객의 생존 확률은 **{survival_prob * 100:.1f}%** 입니다.")
    
    if survival_prob > 0.5:
        st.balloons() # 성공 시 풍선 효과 (재미 요소)