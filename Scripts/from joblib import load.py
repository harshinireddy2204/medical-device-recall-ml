from joblib import load
model = load("rf_model_v1.joblib")
print(model.feature_names_in_)
