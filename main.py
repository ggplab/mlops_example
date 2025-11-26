import pickle
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any # 타입 힌트를 위해 Any를 추가했습니다.

# ----------------------------------------------------
# 1. FastAPI 앱 생성
# ----------------------------------------------------
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="머신러닝 예측 서비스 예제"
)

# ----------------------------------------------------
# 2. Pydantic 모델 정의
# ----------------------------------------------------
# 클라이언트로부터 받을 입력 데이터의 형태를 정의
class PredictionInput(BaseModel):
    sex: str # male 또는 female
    age: float

# ----------------------------------------------------
# 3. 모델 로드 (서버 시작 시 한 번만 실행)
# ----------------------------------------------------
MODEL_PATH = "titanic_model.pkl"
model: Any = None
label_encoder: Any = None

if not os.path.exists(MODEL_PATH):
    # 모델 파일이 없으면 서버 시작 불가
    # 실제 강의에서는 이전에 train_model.py를 실행해야 함을 강조
    raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")

try:
    with open(MODEL_PATH, "rb") as f:
        # pickle 파일에서 모델과 레이블 인코더 로드
        model, label_encoder = pickle.load(f)
    print(f"모델 로드 완료 및 준비됨.")
except Exception as e:
    raise RuntimeError(f"모델 로드 중 오류 발생: {e}")


# ----------------------------------------------------
# 4. API 엔드포인트 정의 및 예측 로직 구현
# ----------------------------------------------------

# 기본 루트 엔드포인트
@app.get("/", summary="루트 엔드포인트")
def read_root():
    return {"message": "Welcome to Titanic Survival Prediction API! Check /docs for details."}

# 예측 엔드포인트 (POST)
@app.post("/predict", summary="생존 확률 예측")
def predict(input_data: PredictionInput):
    """
    성별(sex)과 나이(age)를 입력받아 생존 확률을 예측합니다.
    """
    try:
        # 1. 입력 데이터 전처리 (성별 인코딩)
        input_sex = input_data.sex.lower()
        
        # 유효성 검사 (label_encoder에 정의된 값인지 확인)
        if input_sex not in label_encoder.classes_:
            raise HTTPException(
                status_code=400, 
                detail=f"유효하지 않은 성별 값입니다: {input_data.sex}. 'male' 또는 'female'을 입력해주세요."
            )
            
        # LabelEncoder를 사용하여 'male' 또는 'female'을 숫자로 변환 (e.g., male=1, female=0)
        sex_encoded = label_encoder.transform([input_sex])[0]
        
        # 2. 모델 예측 수행
        # numpy 배열 형태로 입력 데이터를 준비: [[성별_인코딩, 나이]]
        X_input = np.array([[sex_encoded, input_data.age]])
        
        # 생존 확률 (1일 확률) 추출
        # predict_proba의 두 번째 요소 ([0][1])가 생존 확률을 나타냅니다.
        survival_prob = model.predict_proba(X_input)[0][1]

        # 3. 결과 반환
        return {
            "sex": input_data.sex,
            "age": input_data.age,
            "survival_probability": round(survival_prob, 3)
        }
        
    except HTTPException as e:
        # 400 에러 발생 시 그대로 반환
        raise e
    except Exception as e:
        # 그 외 내부 오류 발생 시 500 에러 반환
        print(f"예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="예측 중 서버 내부 오류가 발생했습니다.")


# ----------------------------------------------------
# 5. 서버 실행 가이드 (터미널 명령어)
# ----------------------------------------------------
# uvicorn main:app --reload --host 0.0.0.0 --port 8080