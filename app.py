from fastapi import FastAPI

# FastAPI 앱 생성
app = FastAPI()

# 루트 URL 접속 시 실행되는 함수 (GET 메서드)
@app.get("/")
def read_root():
    return {"Hello": "FastAPI World!"}

# 실행: uvicorn main:app --reload --host 0.0.0.0 --port 8080