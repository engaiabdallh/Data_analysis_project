from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

def modeling_classifier(df, target_col='survived', drop_cols=None):
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    X = df.drop(columns=[target_col], axis=1)
    y = df[target_col]

    if y.dtype not in ['int64', 'int32']:
        y = y.round().astype(int)

    X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled_ts = scaler.fit_transform(X_train_ts)
    X_test_scaled_ts = scaler.transform(X_test_ts)

    X_train_scaled_kf = scaler.fit_transform(X_train_kf)#-----
    X_test_scaled_kf = scaler.transform(X_test_kf)

    model=LogisticRegression()
    model.fit(X_train_scaled_kf, y_train_kf)
    y_pred_kf = model.predict(X_test_scaled_kf)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    gbc = GradientBoostingClassifier()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled_ts, y_train_ts)
    rf_best = grid_search.best_estimator_

    voting = VotingClassifier(estimators=[
        ('rf', rf_best),
        ('xgb', xgb),
        ('gbc', gbc)
    ], voting='soft')

    voting.fit(X_train_scaled_ts, y_train_ts)

    y_pred_ts = voting.predict(X_test_scaled_ts)

    accuracy_ts = accuracy_score(y_test_ts, y_pred_ts)
    precision_ts = precision_score(y_test_ts, y_pred_ts, zero_division=0)
    recall_ts = recall_score(y_test_ts, y_pred_ts, zero_division=0)
    f1_ts = f1_score(y_test_ts, y_pred_ts, zero_division=0)
    report_ts = classification_report(y_test_ts, y_pred_ts, zero_division=0, output_dict=True)
    cm_ts = confusion_matrix(y_test_ts, y_pred_ts)

    accuracy_kf = accuracy_score(y_test_kf, y_pred_kf)
    precision_kf = precision_score(y_test_kf, y_pred_kf, zero_division=0)
    recall_kf = recall_score(y_test_kf, y_pred_kf, zero_division=0)
    f1_kf = f1_score(y_test_kf, y_pred_kf, zero_division=0)
    report_kf = classification_report(y_test_kf, y_pred_kf, zero_division=0, output_dict=True)
    cm_kf = confusion_matrix(y_test_kf, y_pred_kf)

    return {
        "train_test_split": {
            "accuracy": accuracy_ts,
            "precision": precision_ts,
            "recall": recall_ts,
            "f1_score": f1_ts,
            "classification_report": report_ts,
            "confusion_matrix": cm_ts
        },
        "kfold_split": {
            "accuracy": accuracy_kf,
            "precision": precision_kf,
            "recall": recall_kf,
            "f1_score": f1_kf,
            "classification_report": report_kf,
            "confusion_matrix": cm_kf
        }
    }
