import pandas as pd
# import numpy as np
# import wfdb
# import ast

path = r"D:/project/arrhythmia-detection-using-1d-cnn/dataset/"
sampling_rate=100

# load and convert annotation data
df = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
# Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
print(df.head())
print(df.info())

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
# agg_df = agg_df[agg_df.diagnostic == 1]
print(agg_df.head())

