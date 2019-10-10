import cufflinks as cf
from datetime import datetime
import numpy as np
from operator import itemgetter
import plotly.offline as offline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

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


def parameter_search(X, y, method, params, random_seed):
    random_seed
    valid = {'elastic', 'rf', 'xgboost', 'stack'}
    if name not in valid:
        raise ValueError("Name must be one of %r." % valid)
        
    if name == 'elastic':
        if not type(params) == list:
            raise ValueError('params is not a dictionary, please support a dictionary of parameters')
        else:
            params = params
            score = []
            end = np.floor(0.8 * X.shape[0]).astype(int)
            X_train = np.array(X)[:end]
            y_train = np.array(y)[:end]
            X_cv = np.array(X)[end:]
            y_cv = np.array(y)[end:]

        # Grid search for elastic net
        for ratio in  params['ratio']:
            for alpha in params['alpha']:
                elastic = ElasticNet(random_state=None, alpha=alpha, l1_ratio=ratio)
                elastic.fit(X_train, y_train)
                preds = elastic.predict(X_cv)                                       
                score.append((alpha, ratio, adam_metric(y_cv, preds)))

                # Get best score
                best_params = min(score, key=itemgetter(2))[0:2]
        

    if name == 'rf':
        if not type(params) == dict:
            raise ValueError('params is not a dictionary, please support a dictionary of parameters')
        else:
            steps = len(params['n_estimators']) * len(params['max_depth'] * len(params['max_features']))
            
            params = params
            score = []
            end = np.floor(0.8 * X.shape[0]).astype(int)
            X_train = np.array(X)[:end]
            y_train = np.array(y)[:end]
            X_cv = np.array(X)[end:]
            y_cv = np.array(y)[end:]

            # Grid search for elastic net
            for step in range(steps):
                n_estimators = np.random.choice(params['n_estimators'])
                max_depth = np.random.choice(params['max_depth'])
                max_features = np.random.choice(params['max_features'])

                rf = RandomForestRegressor(n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                max_features=max_features)
                rf.fit(X_train, y_train)
                preds = rf.predict(X_cv)                                       
                score.append((n_estimators, max_depth, max_features, adam_metric(y_cv, preds)))

            # Get best score
            best_params = min(score, key=itemgetter(3))[0:3]
        
    
    return best_params
    
    
    
    
    