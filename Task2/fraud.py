import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\codsoft tasks\credit card fraud detection\fraudTest.csv.zip")

print("Original Shape:", df.shape)

df_fraud = df[df['is_fraud'] == 1]
df_normal = df[df['is_fraud'] == 0].sample(20000, random_state=42)

df = pd.concat([df_fraud, df_normal])

print("After Sampling:", df.shape)
print(df['is_fraud'].value_counts())

df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['age'] = 2026 - df['dob'].dt.year

drop_cols = ['first', 'last', 'cc_num', 'trans_date', 'street', 'dob']
df = df.drop(columns=drop_cols, errors='ignore')

df = pd.get_dummies(df, drop_first=True)

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))