import pandas as pd

data = {
    "income": [40000, 50000, 60000, 70000, 80000],
    "credit_score": [650, 700, 750, 800, 850],
    "gpa": [2.5, 3.0, 3.5, 4.0, 3.8],
    "debt_to_income_ratio": [0.3, 0.2, 0.25, 0.15, 0.1],
    "loan_approval": [0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv("student_loans.csv", index=False)
print("Dummy dataset saved successfully!")
