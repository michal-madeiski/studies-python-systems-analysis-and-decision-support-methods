import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, accuracy_score
from part_1.main_code.data_loader import loaded_fifa22_data as data
from part_2.linear_regr.lin_regr_closed_form import LinearRegressionClosedFormula


#print(data.info())
#print(data.describe(include="all"))

num_models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "SVM": SVR(kernel="rbf", C=1.0),
    "GradientBoosting": GradientBoostingRegressor(max_depth=5, random_state=42, learning_rate=0.5, subsample=0.8),
    "LinearRegressionClosedFormula": LinearRegressionClosedFormula(),
}

cat_models = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

features = ["overall", "value_eur", "age", "height_cm", "weight_kg", "league_level", "club_position", "preferred_foot", "weak_foot", "skill_moves"]
target = "overall"

data = data[features]

#print("Missing values per column:\n", data.isnull().sum())

X = data.drop(target, axis=1)
y = data[target]

X = X[~y.isnull()]
y = y.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

mode = "cat"
models = cat_models
if pd.api.types.is_numeric_dtype(y):
    models = num_models
    mode = "num"

models_fit = {}
for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    clf.fit(X_train, y_train)
    models_fit[name] = clf

results = {}
for name, clf in models_fit.items():
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred) if mode == "cat" else r2_score(y_test, y_pred)
    results[name] = score


if __name__ == "__main__":
    for k, v in results.items():
        print(f"{k}: {v:.3f}")