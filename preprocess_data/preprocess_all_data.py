import os

for name in ['wikipedia', 'reddit', 'enron','uci', 'CanParl']:
    os.system(f'python preprocess_data.py  --dataset_name {name}')
