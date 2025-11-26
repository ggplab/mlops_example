import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 데이터 불러오기
df = sns.load_dataset("titanic")

# 필요한 컬럼 선택 (Sex, Age, Survived)
df = df[["sex", "age", "survived"]].dropna()

# 데이터 전처리 (Sex: Label Encoding, Age: 결측값 평균 대체)
label_encoder = LabelEncoder()
df["sex"] = label_encoder.fit_transform(df["sex"])  # male=1, female=0

# X, y 정의
X = df[["sex", "age"]]
y = df["survived"]

# 모델 생성 및 학습 (Logistic Regression)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # Age 결측값 처리
    ("classifier", LogisticRegression())
])
model.fit(X, y)

# 모델 저장 (pickle)
with open("titanic_model.pkl", "wb") as f:
    pickle.dump((model, label_encoder), f)

print("모델 학습 완료 & 저장됨: titanic_model.pkl")