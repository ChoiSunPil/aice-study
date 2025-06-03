
# 📅 Day 2 — Tabular 데이터 처리 실습

## 🎯 목표
- Pandas를 활용한 데이터 로딩 및 탐색
- 결측치, 이상치, 범주형 처리 등 전처리 기법 실습
- 시각화를 통해 데이터 구조 이해
- 모델 학습을 위한 데이터 정제 기반 마련

---

## 📘 이론 개념 요약

### ✅ Tabular 데이터란?
- 열(Column)과 행(Row)으로 구성된 구조화된 테이블 형식의 데이터
- 예: 엑셀, CSV, 관계형 DB 테이블 등

### ✅ 주요 전처리 작업
| 유형 | 설명 |
|------|------|
| 결측치 처리 | `dropna()`, `fillna()` |
| 이상치 탐지 | Z-score, IQR |
| 범주형 인코딩 | Label Encoding, One-hot Encoding |
| 정규화 | `MinMaxScaler`, `StandardScaler` |
| 데이터 분리 | 학습/검증/테스트 세트 분리 |

---

## 💻 실습 과제

### 🔹 데이터셋: Titanic (from Kaggle)
- 다운로드: [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
- `train.csv` 파일을 `data/` 디렉토리에 저장

---

### ✅ 실습 1: EDA (Exploratory Data Analysis)

```python
import pandas as pd

df = pd.read_csv("../data/train.csv")
print(df.head())
print(df.info())
print(df.describe())
```

- [ ] 열 별 데이터 타입 확인
- [ ] `isnull().sum()`으로 결측치 확인
- [ ] 범주형 변수 확인: `Sex`, `Embarked`, `Pclass`

---

### ✅ 실습 2: 결측치 처리

```python
# Age 결측치는 평균으로 대체
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Embarked 결측치는 최빈값으로 대체
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

---

### ✅ 실습 3: 범주형 인코딩

```python
# Label Encoding (Sex)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot Encoding (Embarked)
df = pd.get_dummies(df, columns=['Embarked'])
```

---

### ✅ 실습 4: 이상치 탐지 (IQR 방식)

```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 제거
df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]
```

---

### ✅ 실습 5: 정규화

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

---

### ✅ 실습 6: 학습용 데이터 구성

```python
from sklearn.model_selection import train_test_split

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 📊 실습 7: 시각화 (선택 과제)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

sns.boxplot(x='Pclass', y='Age', data=df)
```

---

## 🧾 마무리 체크리스트

| 항목 | 완료 여부 |
|------|-----------|
| Titanic 데이터셋 다운로드 | ☐ |
| Pandas로 데이터 로딩 및 요약 확인 | ☐ |
| 결측치 및 이상치 처리 완료 | ☐ |
| 범주형 인코딩 완료 | ☐ |
| 정규화 및 데이터 분리 완료 | ☐ |
| 시각화로 탐색 완료 (선택) | ☐ |

---

## 📝 참고 자료
- [Pandas 공식 문서](https://pandas.pydata.org/docs/)
- [Seaborn 시각화 가이드](https://seaborn.pydata.org/examples/index.html)
- [Kaggle Titanic EDA 예시](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)
