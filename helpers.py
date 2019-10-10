#!/home/stas/anaconda3/envs/comp1/bin/python

import cufflinks as cf
from datetime import datetime
import numpy as np
import plotly.offline as offline
import pandas as pd

def date_convert(df):
    if 'Date' in df.columns.tolist():
        df.loc[:,'Date'] = pd.to_datetime(df['Date'])
        
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.week
        df['Day'] = df['Date'].dt.day
        
        #df = df.drop(axis=1, labels='Date')
        
        cols_after = df.columns.tolist()[0:9]
        cols_before = df.columns.tolist()[9:]
        df = df.loc[:, cols_before + cols_after]
        
    else:
        raise ValueError('Date column is not found in df')
        
    return df



def plot_hists(data, name):
    valid = {'store', 'train'}
    if name not in valid:
        raise ValueError("Name must be one of %r." % valid)
    
    if name == 'store':
        fig = data.iplot(kind='histogram', subplots=True, shared_yaxes=False, shape=(2,5), asFigure=True, theme='solar')
        
    if name == 'train':
        print('The data frame is too big, so only 50K examples are taken into account for the histogram')
        
        ind = np.random.randint(0, data.shape[0], size=50000)
        data = data.iloc[ind]
        fig = data.iplot(kind='histogram', subplots=True, shared_yaxes=False, shape=(2,7), asFigure=True, theme='solar')
         
    return offline.iplot(fig)
        
        
        
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
        
        