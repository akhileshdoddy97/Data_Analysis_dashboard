import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Stock Mispricing Classifier")

st.write("Machine Learning model to classify stocks as Overvalued or Undervalued.")

# load dataset
data = pd.read_csv("master_dataset.csv")

# remove text columns
drop_cols = [
    "ticker",
    "company_name",
    "cik",
    "sector",
    "quarter_end_date",
    "fiscal_quarter"
]

data = data.drop(columns=drop_cols)

# split X and y
X = data.drop(columns=["valuation_label"])
y = data["valuation_label"]

# train model
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X, y)

st.subheader("Select Observation")

row_index = st.selectbox("Choose dataset row", X.index)

input_data = X.loc[[row_index]]

prediction = model.predict(input_data)[0]

if prediction == 1:
    result = "Overvalued"
else:
    result = "Undervalued"

st.write("### Prediction:", result)