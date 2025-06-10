import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer # mean=0,std=1
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def preprocess_titanic(df, display_log: bool = False):
    logs = []
    log_dict = {
        "dropped_columns": [],
        "imputed_numeric": [],
        "imputed_categorical": [],
        "encoded": [],
        "scaled": [],
        "onehot_encoded": [],
        "outliers_handled": []
    }

    logs.append("Starting preprocessing...")

    threshold = 0.5 * len(df)
    dropped_cols = df.columns[df.isnull().sum() >= threshold].tolist()
    df = df.dropna(thresh=threshold, axis=1)
    if dropped_cols:
        logs.append(f"Dropped columns with ≥50% missing: {', '.join(dropped_cols)}")
        log_dict["dropped_columns"].extend(dropped_cols)
    else:
        logs.append("No columns dropped due to missingness.")

    imputer_num = SimpleImputer(strategy='median')    #-----
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
        logs.append(f"Imputed numeric columns: {', '.join(numeric_cols)}")
        log_dict["imputed_numeric"].extend(numeric_cols)

    imputer_cat = SimpleImputer(strategy='most_frequent')
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        logs.append(f"Imputed categorical columns: {', '.join(categorical_cols)}")
        log_dict["imputed_categorical"].extend(categorical_cols)

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outliers > 0:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            logs.append(f"Handled {outliers} outliers in '{col}' using IQR method.")
            log_dict["outliers_handled"].append(col)

    if 'sibsp' in df.columns and 'parch' in df.columns:
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        df['is_child']  = (df['age'] < 12).astype(int)
        df['is_alone']  = (df['family_size'] == 1).astype(int)
        df['age_group'] = pd.cut(df['age'],
                                 bins=[0,12,18,35,60,100],
                                 labels=[0,1,2,3,4]).astype(int)
        logs.append("Engineered: family_size, is_child, is_alone, age_group")

    if 'fare' in df.columns:    #---------
        df['fare_log'] = np.log1p(df['fare'])
        logs.append("Engineered: fare_log")

    to_onehot = [c for c in ['embarked','embark_town'] if c in df.columns]
    if to_onehot:
        onh = OneHotEncoder(sparse_output=False, drop='first')
        enc = onh.fit_transform(df[to_onehot])
        enc_df = pd.DataFrame(enc,
                              columns=onh.get_feature_names_out(to_onehot),
                              index=df.index)
        df = df.drop(columns=to_onehot + ['alive'], errors='ignore')
        df = pd.concat([df, enc_df], axis=1)
        logs.append(f"One-hot encoded: {', '.join(to_onehot)}")
        log_dict["onehot_encoded"].extend(to_onehot)

    for col in ['class', 'who', 'sex', 'adult_male']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    logs.append("Label-encoded: class, who, sex, adult_male")
    log_dict["encoded"].extend(['class', 'who', 'sex', 'adult_male'])

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logs.append(f"Standardized numeric columns: {', '.join(numeric_cols)}")
        log_dict["scaled"].extend(numeric_cols)

    logs.append("Preprocessing completed.")

    if display_log:
        print("\nLogs:")
        for log in logs:
            print(log)
        print("\nStructured Log (Dictionary):")
        print(log_dict)

    return df, log_dict
