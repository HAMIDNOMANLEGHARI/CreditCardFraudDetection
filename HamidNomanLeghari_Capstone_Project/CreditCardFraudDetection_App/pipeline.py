import pandas as pd
import numpy as np

def preprocess_data(df):
    df_clean = df.copy()

    # 1. Datetime Features
    if 'TransactionDT' in df_clean.columns:
        df_clean['TransactionDT_parsed'] = pd.to_datetime(df_clean['TransactionDT'], unit='s')
        df_clean['DT_hour'] = df_clean['TransactionDT_parsed'].dt.hour
        df_clean['DT_day'] = df_clean['TransactionDT_parsed'].dt.day
        df_clean['DT_weekday'] = df_clean['TransactionDT_parsed'].dt.weekday
    
    # 2. Time Difference Features
    if 'card1' in df_clean.columns and 'TransactionDT_parsed' in df_clean.columns:
        df_clean = df_clean.sort_values(['card1', 'TransactionDT_parsed'])
        df_clean['time_diff'] = df_clean.groupby('card1')['TransactionDT_parsed'].diff().dt.total_seconds()
        df_clean['time_diff'] = df_clean['time_diff'].fillna(-1)

    # 3. Aggregation Features
    agg_funcs = ['count', 'mean', 'std']
    for col in ['card1', 'card2']:
        if col in df_clean.columns:
            for func in agg_funcs:
                df_clean[f'{col}_TransactionAmt_{func}'] = df_clean.groupby(col)['TransactionAmt'].transform(func)

    # 4. Browser & Resolution Parsing
    if 'id_33' in df_clean.columns:
        resolution_split = df_clean['id_33'].astype(str).str.split('x', expand=True)
        if resolution_split.shape[1] == 2: 
            df_clean['screen_width'] = resolution_split[0].astype(float)
            df_clean['screen_height'] = resolution_split[1].astype(float)
        df_clean = df_clean.drop('id_33', axis=1)

    if 'id_31' in df_clean.columns:
        df_clean['browser_clean'] = df_clean['id_31'].astype(str).str.lower()
        major_browsers = ['chrome', 'safari', 'firefox', 'edge', 'ie', 'samsung']
        for browser in major_browsers:
            df_clean.loc[df_clean['browser_clean'].str.contains(browser, na=False), 'browser_clean'] = browser
        df_clean = df_clean.drop('id_31', axis=1)

    # 5. UID Creation
    if all(c in df_clean.columns for c in ['card1', 'card2']):
        df_clean['uid'] = df_clean['card1'].astype(str) + '_' + df_clean['card2'].astype(str)
        for func in agg_funcs:
            df_clean[f'uid_TransactionAmt_{func}'] = df_clean.groupby('uid')['TransactionAmt'].transform(func)
        df_clean = df_clean.drop(['uid'], axis=1)

    # 6. Null Handling
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(-999)

    obj_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[obj_cols] = df_clean[obj_cols].fillna('missing')

    # 7. Categorical Encoding
    low_card_thresh = 10
    for col in obj_cols:
        if df_clean[col].nunique() <= low_card_thresh:
            dummies = pd.get_dummies(df_clean[col], prefix=col, dummy_na=True)
            df_clean = pd.concat([df_clean, dummies], axis=1)
            df_clean = df_clean.drop(col, axis=1)
        else:
            freq = df_clean[col].value_counts() / len(df_clean)
            df_clean[col] = df_clean[col].map(freq)

    # 8. Drop unnecessary columns
    cols_to_drop = ['isFraud', 'TransactionID', 'TransactionDT', 'TransactionDT_parsed', 'TransactionDT_Real']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')

    return df_clean
