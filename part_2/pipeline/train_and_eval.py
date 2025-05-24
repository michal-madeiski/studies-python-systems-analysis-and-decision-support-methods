import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, accuracy_score
from part_1.main_code.data_loader import loaded_fifa22_data as data
from part_2.linear_regr.lin_regr_closed_form import LinearRegressionClosedFormula
from part_2.linear_regr.lin_regr_grad_desc import LinearRegressionGradientDescent

#print(data.info())
#print(data.describe(include="all"))

rd = 42
alpha_L1 = 0.1
alpha_L2 = 10
hyper_param_tuning_flag = True

num_models = {
    "LinearRegression": LinearRegression(),
    "Lasso (L1)": Lasso(alpha=alpha_L1),
    "Ridge (L2)": Ridge(alpha=alpha_L2),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=rd),
    "SVM": SVR(kernel="rbf", C=1.0),
    "GradientBoosting": GradientBoostingRegressor(max_depth=5, random_state=rd, learning_rate=0.1, subsample=0.8),
    "LinearRegressionClosedFormula": LinearRegressionClosedFormula(),
    "LinearRegressionGradientDescent": LinearRegressionGradientDescent(learning_rate=0.01, max_iterations=100, batch_size=32, random_state=rd),
    "LinearRegressionGradientDescent_L1": LinearRegressionGradientDescent(learning_rate=0.01, max_iterations=100, batch_size=32, l1_reg=alpha_L1, random_state=rd),
    "LinearRegressionGradientDescent_L2": LinearRegressionGradientDescent(learning_rate=0.01, max_iterations=100, batch_size=32, l2_reg=alpha_L2, random_state=rd),
}
l1_l2_comp_models = ["LinearRegression", "Lasso (L1)", "Ridge (L2)", "LinearRegressionGradientDescent", "LinearRegressionGradientDescent_L1", "LinearRegressionGradientDescent_L2"]
param_SVM = {
    "kernel": ["rbf", "poly", "sigmoid"],
    "C": [0.01, 0.1, 1.0, 10.0],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

cat_models = {
    "RandomForestClassifier": RandomForestClassifier(random_state=rd),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=rd),
    "SVM": SVC(kernel="rbf", C=1.0),
    "GradientBoosting": GradientBoostingClassifier(random_state=rd),
}
param_DecisionTreeClassifier = {
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [2, 4, 8],
    "criterion": ["gini", "entropy"],
}

features = ["overall", "value_eur", "age", "height_cm", "weight_kg", "league_level", "club_position", "preferred_foot", "weak_foot", "skill_moves"]
targets = ["overall", "value_eur", "age", "club_position", "preferred_foot"]

data = data[features]

num_models_arr = [k for k, _ in num_models.items()]
df_num_comp = pd.DataFrame({"model": num_models_arr})
df_num_comp_train = pd.DataFrame({"model": num_models_arr})
df_num_comp_diff = pd.DataFrame({"model": num_models_arr})
df_num_comp_l1_l2 = pd.DataFrame({"model": num_models_arr})
all_mse = []
cat_models_arr = [k for k, _ in cat_models.items()]
df_cat_comp = pd.DataFrame({"model": cat_models_arr})
df_cat_comp_train = pd.DataFrame({"model": cat_models_arr})
df_cat_comp_diff = pd.DataFrame({"model": cat_models_arr})

#print("Missing values per column:\n", data.isnull().sum())

cv_splits = 3
num_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=rd)
cat_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=rd)
df_num_cv_comp = pd.DataFrame({"model": num_models_arr})
df_cat_cv_comp = pd.DataFrame({"model": cat_models_arr})

for target in targets:
    #print(target) #for coef comp in lin_reg w/ and w/o regularization
    X = data.drop(target, axis=1)
    y = data[target]

    X = X[~y.isnull()]
    y = y.dropna()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        #("poly", PolynomialFeatures(degree=3, include_bias=False)), #comment non-linear num_models and saving their results; uncomment saving poly results
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ])

    mode = "cat"
    models = cat_models
    cv = cat_cv
    scoring = "accuracy"
    if pd.api.types.is_numeric_dtype(y):
        models = num_models
        mode = "num"
        cv = num_cv
        scoring = "r2"

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.fit_transform(X_test)

    models_fit = {}
    if not hyper_param_tuning_flag:
        for name, model in models.items():
            models_fit[name] = model.fit(X_train_transformed, y_train)

    if hyper_param_tuning_flag:
        if mode == "num"  and target == "value_eur":
            grid_SVM = GridSearchCV(num_models["SVM"], param_SVM, cv=cv, scoring=scoring)
            grid_SVM.fit(X_train_transformed, y_train)
            print(f"Target: {target} | Best params: {grid_SVM.best_params_} | Best CV score: {grid_SVM.best_score_:.3f}")
            models_fit["SVM"] = grid_SVM.best_estimator_
        if mode == "cat"  and target == "club_position":
            grid_DecisionTreeClassifier = GridSearchCV(cat_models["DecisionTree"], param_DecisionTreeClassifier, cv=cv, scoring=scoring)
            grid_DecisionTreeClassifier.fit(X_train_transformed, y_train)
            print(f"Target: {target} | Best params: {grid_DecisionTreeClassifier.best_params_} | Best CV score: {grid_DecisionTreeClassifier.best_score_:.3f}")
            models_fit["DecisionTree"] = grid_DecisionTreeClassifier.best_estimator_

    results = {}
    for name, clf in models_fit.items():
        y_pred = clf.predict(X_test_transformed)
        y_pred_train = clf.predict(X_train_transformed)
        score = accuracy_score(y_test, y_pred) if mode == "cat" else r2_score(y_test, y_pred)
        score_train = accuracy_score(y_train, y_pred_train) if mode == "cat" else r2_score(y_train, y_pred_train)
        results[name] = (score, score_train)

        #coef comp in lin_reg w/ and w/o regularization
        #if mode == "num":
        #     print(f"{name}: {clf.coef_}")

    results_cv = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train_transformed, y_train, cv=cv, scoring=scoring)
        scores_rounded = [f"{x:.3f}" for x in scores]
        results_cv[name] = f"scores: {scores_rounded} | mean: {scores.mean():.3f} | std: {scores.std():.3f}"

    for k, v in results.items():
        #print(f"{k}: Train={v[1]:.3f}; Test={v[0]:.3f}")
        result = [v[0] for _, v in results.items()]
        result_train = [v[1] for _, v in results.items()]
        result_diff = [abs(v[0] - v[1]) for _, v in results.items()]
        if mode == "cat":
            df_cat_comp[target] = result
            df_cat_comp_train[target] = result_train
            df_cat_comp_diff[target] = result_diff
        if mode == "num":
            df_num_comp[target] = result
            df_num_comp_train[target] = result_train
            df_num_comp_diff[target] = result_diff
            if k in l1_l2_comp_models:
                df_num_comp_l1_l2[target] = result

    if mode == "num":
        all_mse.append([target] + num_models["LinearRegressionGradientDescent"].all_mse)

    for k, v in results_cv.items():
        #print(f"{k}: {v}")
        result = [v for _, v in results_cv.items()]
        if mode == "cat":
            df_cat_cv_comp[target] = result
        if mode == "num":
            df_num_cv_comp[target] = result

df_cat_comp.to_csv("../comparison/cat_comp.csv", index=False, float_format="%.3f")
df_num_comp.to_csv("../comparison/num_comp.csv", index=False, float_format="%.3f")
#df_num_comp.to_csv("../comparison/num_comp_poly.csv", index=False, float_format="%.3f")
df_cat_cv_comp.to_csv("../comparison/cat_cv_comp.csv", index=False, float_format="%.3f")
df_num_cv_comp.to_csv("../comparison/num_cv_comp.csv", index=False, float_format="%.3f")
#df_num_cv_comp.to_csv("../comparison/num_cv_comp_poly.csv", index=False, float_format="%.3f")
df_cat_comp_train.to_csv("../comparison/cat_comp_train.csv", index=False, float_format="%.3f")
df_num_comp_train.to_csv("../comparison/num_comp_train.csv", index=False, float_format="%.3f")
#df_num_comp_train.to_csv("../comparison/num_comp_train_poly.csv", index=False, float_format="%.3f")
df_cat_comp_diff.to_csv("../comparison/cat_comp_diff.csv", index=False, float_format="%.3f")
df_num_comp_diff.to_csv("../comparison/num_comp_diff.csv", index=False, float_format="%.3f")
#df_num_comp_diff.to_csv("../comparison/num_comp_diff_poly.csv", index=False, float_format="%.3f")
df_num_comp_l1_l2.to_csv("../comparison/num_comp_l1_l2.csv", index=False, float_format="%.3f")

df_num_comp_mse = pd.DataFrame(all_mse, columns=["target"] + [f"e{i+1}" for i in range(len(all_mse[0][1:]))])
df_num_comp_mse.to_csv("../comparison/all_mse.csv", index=False, float_format="%.3f")
#df_num_comp_mse.to_csv("../comparison/all_mse_poly.csv", index=False, float_format="%.3f")