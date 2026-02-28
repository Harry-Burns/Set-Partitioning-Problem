import numpy as np

def load_sppwn(f_name):
    file_path = f'data/{f_name}.txt'

    with open(file_path, 'r') as f:
        data = f.readlines()

    data = [[int(x) for x in d.strip().split()] for d in data]
    
    N_ROWS,N_COLS = data[0]

    COL_COSTS = np.array([d[0] for d in data[1:]])
    COL_ROWS = np.array([np.isin(np.arange(1,N_ROWS+1), d[2:]) for d in data[1:]])

    if f_name == 'sppnw41': TARGET_OPTIMAL = 11307
    elif f_name == 'sppnw42': TARGET_OPTIMAL = 7656
    elif f_name == 'sppnw43': TARGET_OPTIMAL = 8904
    else: TARGET_OPTIMAL = 0

    return (N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL