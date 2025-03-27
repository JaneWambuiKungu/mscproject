import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


data = {
    "income": [40000, 50000, 60000, 70000, 80000],
    "credit_score": [650, 700, 750, 800, 850],
    "gpa": [2.5, 3.0, 3.5, 4.0, 3.8],
    "loan_approval": [0, 1, 1, 1, 1]  # Target variable
}

df = pd.DataFrame(data)


FEATURES = ["income", "credit_score", "gpa"]

X = df[FEATURES]
y = df["loan_approval"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Scale features (only 3 columns)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)

# ✅ Train Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# ✅ Save models
pickle.dump(dt_model, open("dt_model.pkl", "wb"))
pickle.dump(lr_model, open("lr_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Models trained with 3 features!")
