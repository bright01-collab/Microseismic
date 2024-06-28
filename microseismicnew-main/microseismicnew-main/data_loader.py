import pandas as pd

def load_data(dataset, start_date, end_date):
    if dataset == "SLB Data":
        data = pd.read_csv('SLB_Data.csv')
        origin_time_col = 'SLB origin time'
    elif dataset == "Relocated Data":
        data = pd.read_csv('Relocated_Data.csv')
        origin_time_col = 'Relocated origin time'
    
    data[origin_time_col] = pd.to_datetime(data[origin_time_col], format='%d/%m/%Y %H:%M:%S.%f')

    data = data[(data[origin_time_col] >= start_date) & (data[origin_time_col] <= end_date)]

    return data

