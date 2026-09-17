import os
import re
import numpy as np
import pandas as pd
import kagglehub 
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ==========================================
# 1. kagglehub를 통한 데이터 다운로드 및 경로 로드
# ==========================================
print("--- 데이터 다운로드 시작 ---")
competition_path = kagglehub.competition_download('titanic')
print("타이타닉 대회 파일 다운로드 완료 경로:", competition_path)

train = pd.read_csv(os.path.join(competition_path, 'train.csv'))       
test = pd.read_csv(os.path.join(competition_path, 'test.csv'))         
submission = pd.read_csv(os.path.join(competition_path, 'gender_submission.csv')) 


# ==========================================
# 2. 데이터 전처리 (결측치 처리)
# ==========================================
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train['Embarked'] = train['Embarked'].fillna('S')


# ==========================================
# 3. 데이터 전처리 (기존 범주형 변수 수치 변환)
# ==========================================
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train['Embarked'] = train['Embarked'].map({'S': 2, 'C': 1, 'Q': 0})
test['Embarked'] = test['Embarked'].map({'S': 2, 'C': 1, 'Q': 0})


# ==========================================
# 4. Feature Engineering (파생 변수 생성)
# ==========================================
datasets = [train, test]

for df in datasets:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0).astype(int)


# ==========================================
# 5. 학습할 피처(X)와 타겟(Y) 정의
# ==========================================
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
target = 'Survived'

x = train[features]
y = train[target]
x_test = test[features]


# ==========================================
# 6. Stratified K-Fold 검증 및 LightGBM 학습
# ==========================================
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

oof_preds = np.zeros(len(train)) 
test_preds = np.zeros(len(test)) 

print("\n--- Stratified K-Fold 교차 검증 시작 ---")

for fold, (train_idx, val_idx) in enumerate(folds.split(x, y)):
    x_train_fold, y_train_fold = x.iloc[train_idx], y.iloc[train_idx]
    x_val_fold, y_val_fold = x.iloc[val_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(
        random_state=123, 
        max_depth=4, 
        n_estimators=100, 
        learning_rate=0.05,
        verbose=-1
    )
    
    model.fit(x_train_fold, y_train_fold)
    
    val_preds = model.predict_proba(x_val_fold)[:, 1]
    oof_preds[val_idx] = val_preds
    
    val_preds_binary = (val_preds >= 0.5).astype(int)
    fold_acc = accuracy_score(y_val_fold, val_preds_binary)
    print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")
    
    test_preds += model.predict_proba(x_test)[:, 1] / folds.n_splits

oof_preds_binary = (oof_preds >= 0.5).astype(int)
total_acc = accuracy_score(y, oof_preds_binary)
print(f"\n[종합 교차검증 점수] Overall OOF Accuracy: {total_acc:.4f}")


# ==========================================
# 7. 정답지 작성 및 직관적인 생존 여부 라벨 추가
# ==========================================
final_predictions = (test_preds >= 0.5).astype(int)

# 제출용 데이터프레임에는 규격 규칙대로 0과 1을 넣습니다.
submission['Survived'] = final_predictions

# [추가] 사용자가 눈으로 쉽게 확인하기 위한용도로 '생존_여부' 텍스트 컬럼을 추가합니다.
submission['생존_여부'] = submission['Survived'].map({0: '사망 ❌', 1: '생존 ⭕'})

# 화면에 몇몇이 살아남았는지 예시 데이터와 요약 통계를 출력합니다.
print("\n--- 💡 승객별 생존 여부 예측 결과 (상위 10개 행) ---")
print(submission.head(10).to_string(index=False))

survived_count = np.sum(final_predictions)
dead_count = len(final_predictions) - survived_count
total_count = len(final_predictions)

print(f"\n--- 최종 예측 결과 요약 ---")
print(f"예측 대상자 총 {total_count}명 중:")
print(f"  - 사망 예측: {dead_count}명")
print(f"  - 생존 예측: {survived_count}명")
print(f"최종 예측 생존율: {(survived_count / total_count) * 100:.2f}%")


# ==========================================
# 8. 최종 결과 제출 파일 저장
# ==========================================
# 캐글 대회 규격에는 오직 'PassengerId'와 'Survived' 컬럼만 포함되어야 하므로 
# 위에서 만든 텍스트 라벨 컬럼은 제외하고 기존 양식만 파일로 깔끔하게 내보냅니다.
submission[['PassengerId', 'Survived']].to_csv('submission_kagglehub.csv', index=False)
print("\n제출 규격에 맞춘 'submission_kagglehub.csv' 파일이 성공적으로 저장되었습니다.")

