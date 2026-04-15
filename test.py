import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Generate dummy data
df = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [25, 30, 35, 40, 45],
    'Approved': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

sensitive_attribute = 'Gender'
target_variable = 'Approved'

processed_data = df.copy()
le_target = LabelEncoder()
y = le_target.fit_transform(processed_data[target_variable])
X = processed_data.drop(columns=[target_variable])
# Change here:
sensitive_attribute_values = processed_data[sensitive_attribute]
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X_encoded, y, sensitive_attribute_values, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results_df = pd.DataFrame({
    'Sensitive_Attribute': sens_test,
    'Prediction': predictions
})

group_rates = results_df.groupby('Sensitive_Attribute')['Prediction'].mean()

print(group_rates)
print("Done")
