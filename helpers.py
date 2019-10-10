import cufflinks as cf
from datetime import datetime
import numpy as np
import plotly.offline as offline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

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
        
        
        
def adam_metric(actuals, preds):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / (actuals + 1e-9)) / np.sqrt(preds.shape[0])


# One Hot Encoding for Catergory Columns in Dataframe + removes Original Column afterwards
def cat_to_int(df, columnlist):
    for i in columnlist:
        df = pd.concat([df, pd.get_dummies(df[i], prefix=i)], axis=1)
    df = df.drop(columnlist, axis=1, errors='ignore')   
    return df       

# Transforms to Int-Format
def float_to_int(df, columnlist):
    for i in columnlist:
        df[i] = df[i].astype(int)
    return df

# Convert Year and Weeknumber to Datetime-Format
def year_week(y, w):
    return datetime.strptime(f'{y} {w} 1', '%G %V %u')

'''
def parameter_search(X, y, method, params, random_seed, random=False):
    random_seed
    valid = {'elastic', 'rf', 'xgboost'}
    if name not in valid:
        raise ValueError("Name must be one of %r." % valid)
        
    if name == 'elastic':
        if not type(params) == list:
            raise ValueError('params is not a list, please support a list of parameters')
        else:
            params = params
            score = []

        # Grid search for elastic net
        for i in params:
            X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0, 1e6))
            elastic = ElasticNet(random_state=random_seed, alpha=i)
            elastic.fit(X_train, y_train)
            preds = elastic.predict(X_cv)
            score.append(adam_metric(y_cv, preds))


best_param = params[np.argmax(np.array(score))]

# Refit 
elastic = ElasticNet(random_state=seed, alpha=best_param)
elastic.fit(X, y)
'''    
    
    
    
    
    
    