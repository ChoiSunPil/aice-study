

### ✅ Conda 환경 생성

``` bash
conda create -n aice_env python=3.10 -y
conda activate aice_env
```

### ✅ 필수 라이브러리 설치

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter \
            tensorflow torch torchvision opencv-python nltk konlpy
```



### 📂 실습 폴더 구조 구성

``` bash
aice-study/
├── data/
├── notebooks/
│   ├── 01_tabular_eda.ipynb
│   ├── 02_text_preprocessing.ipynb
│   └── ...
├── models/
└── utils/
```