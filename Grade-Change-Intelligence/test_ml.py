import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/historical_data.csv')
df['target_off_spec'] = df.groupby(['current_grade', 'target_grade'])['off_spec'].shift(-30).fillna(0).astype(int)
df['transition_id'] = df['current_grade'] + '_' + df['target_grade']

# Add rate of change (first difference)
for col in ['speed', 'stock_flow', 'steam', 'filler']:
    df[col+'_rate'] = df.groupby('transition_id')[col].diff().fillna(0)

# Add lags (e.g. lag of 15 seconds)
for col in ['speed', 'stock_flow', 'steam', 'filler']:
    df[col+'_lag'] = df.groupby('transition_id')[col].shift(15).fillna(method='bfill').fillna(0)

tr_ids = df['transition_id'].unique()
tr_train, tr_test = train_test_split(tr_ids, test_size=0.2, random_state=42)

train_df = df[df['transition_id'].isin(tr_train)]
test_df = df[df['transition_id'].isin(tr_test)]

# Test 1: Original feature names, but with shifted label (re-labeling)
feats_orig = ['speed', 'stock_flow', 'steam', 'filler', 'basis_weight', 'moisture', 'ash', 'caliper']
model_orig = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
model_orig.fit(train_df[feats_orig], train_df['target_off_spec'])
preds_orig = model_orig.predict(test_df[feats_orig])
print("Original Features + Shifted Label (30s):")
print(classification_report(test_df['target_off_spec'], preds_orig))

# Test 2: Only manipulated variables + rates + lags (no quality variables at all)
feats_manip = ['speed', 'stock_flow', 'steam', 'filler', 
               'speed_rate', 'stock_flow_rate', 'steam_rate', 'filler_rate',
               'speed_lag', 'stock_flow_lag', 'steam_lag', 'filler_lag']
model_manip = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
model_manip.fit(train_df[feats_manip], train_df['target_off_spec'])
preds_manip = model_manip.predict(test_df[feats_manip])
print("Manipulated Variables + Lags + Rates of Change (no quality variables):")
print(classification_report(test_df['target_off_spec'], preds_manip))
