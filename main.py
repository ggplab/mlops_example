import pandas as pd
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.prophet_loader import load_model
import uvicorn
import os

# 1. FastAPI 앱 생성
app = FastAPI(title="온도 예측 시스템")

# 2. 템플릿 설정 (HTML 파일 경로)
templates = Jinja2Templates(directory="templates")

# 3. 모델 로드 (서버 시작 시 1회 로드)
model, forecast = load_model()

# 예측 가능한 날짜 목록 및 데이터 범위 설정
available_dates = forecast["ds"].dt.date.astype(str).tolist()
actual_data_last_date = pd.to_datetime("2025-02-28")
min_date, max_date = forecast["ds"].min(), forecast["ds"].max()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지 렌더링"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "available_dates": available_dates}
    )

@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request, date: str = Query(...)):
    """날짜를 받아 예측 결과를 보여주는 페이지"""
    try:
        if not date:
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error": "날짜를 선택하세요.", 
                "available_dates": available_dates
            })

        future_date = pd.to_datetime(date)

        # 날짜 범위 체크
        if future_date < min_date or future_date > max_date:
            error_msg = f"예측 가능한 날짜 범위는 {min_date.date()} ~ {max_date.date()}입니다."
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error": error_msg, 
                "available_dates": available_dates
            })

        # 예측 수행
        prediction = forecast[forecast["ds"].dt.date == future_date.date()]
        if prediction.empty:
            return templates.TemplateResponse("index.html", {
                "request": request, 
                "error": "해당 날짜에 대한 예측 데이터가 없습니다.", 
                "available_dates": available_dates
            })

        # 결과 추출
        predicted_temp = round(prediction["yhat"].values[0], 2)
        lower_bound = round(prediction["yhat_lower"].values[0], 2)
        upper_bound = round(prediction["yhat_upper"].values[0], 2)

        # 실제값 확인
        if future_date <= actual_data_last_date:
            actual_temp = round(prediction["y"].values[0], 2)
        else:
            actual_temp = "N/A"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "selected_date": date,
            "predicted_temp": predicted_temp,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "actual_temp": actual_temp,
            "available_dates": available_dates
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": str(e), 
            "available_dates": available_dates
        })

if __name__ == "__main__":
    # 로컬 테스트 용
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)