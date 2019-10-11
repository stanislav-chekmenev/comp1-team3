import cufflinks as cf
from datetime import date, datetime
import numpy as np
from operator import itemgetter
import pandas as pd
import plotly.offline as offline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from tqdm import tqdm



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

def data_transformation(df):
    # convert to Int
    #df = helpers.float_to_int(df, {'Promo2SinceYear', 'Promo2SinceWeek', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'})
    
    # Convert CompetitionYear and CompetitionMonth to datetime format
    df_subset_Comp = df.loc[(~df['CompetitionOpenSinceYear'].isnull()) & (~df['CompetitionOpenSinceMonth'].isnull()), \
                            ['CompetitionOpenSinceYear','CompetitionOpenSinceMonth']]
    df_subset_Comp = float_to_int(df_subset_Comp, {'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'})
    df_subset_Comp['CompetitionStart'] = df_subset_Comp['CompetitionOpenSinceYear'].astype(str) + '-' + \
    df_subset_Comp['CompetitionOpenSinceMonth'].astype(str)  + '-01' 
    df['CompetitionStart'] = pd.to_datetime(df_subset_Comp['CompetitionStart'])
    
    # Convert Promoyear and Promoweekno to datetime format
    df_subset = df.loc[(~df['Promo2SinceYear'].isnull()) & (~df['Promo2SinceWeek'].isnull()), \
                       ['Promo2SinceYear','Promo2SinceWeek']]
    df_subset = float_to_int(df_subset, {'Promo2SinceYear', 'Promo2SinceWeek'})
    df['PromoStart'] = df_subset.apply(lambda row: year_week(row.Promo2SinceYear, row.Promo2SinceWeek), axis=1)

    # create PromoDuration Column:  Date - PromoStart
    df['PromoDuration'] = (df['Date'] - df['PromoStart'])/np.timedelta64(1,'D')
    df['PromoDuration'].fillna(0, inplace=True)
    
    # Calculate is Competition is active and how long the competition is active 
    df['CompetitionActive'] = np.where(df['CompetitionStart'] <= df['Date'], 1, 0)
    df['CompetitionDays'] = (df['Date'] - df['CompetitionStart'])/np.timedelta64(1,'D')
    
    df['RunningAnyPromo'] = 0
    months_abbr = []

    for i in range(1,13):
        months_abbr.append((i, date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (df['PromoInterval'].str.contains(i[1], na=False)) & (df['Month']==i[0]) & (df['Promo2']==1) | df['Promo']==1
        df.loc[mask, 'RunningAnyPromo'] = 1
        
    # Sets RunningPromo to 1 if Months in Substring of PromoIntervall and current month match 
    df['RunningPromo2'] = 0
    months_abbr = []
    for i in range(1,13):
        months_abbr.append((i, date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (df['PromoInterval'].str.contains(i[1], na=False)) & (df['Month']==i[0]) & (df['Promo2']==1)
        df.loc[mask, 'RunningPromo2'] = 1
    df = df.drop({'Date','CompetitionStart','PromoStart'}, axis=1, errors='ignore') 
    
    # Replace NaN with Zeros
    for i in {'Promo2SinceYear', 'Promo2SinceWeek', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Sales'}:
        df[i].fillna(0, inplace=True)
    
    #Replace NaN in Customers with Mean(Customers), but if Store not open set Customers to 0
    df['Customers'].fillna(df['Customers'].mean(), inplace=True)
    df.loc[df['Open'] == 0, 'Customers'] = 0
    
    df = df.drop({'Date','CompetitionStart','PromoStart','PromoInterval','Promo','Promo2','CompetitionDays','DayOfWeek'}, axis=1, errors='ignore') 
    for i in {'Open', 'StateHoliday', 'SchoolHoliday'}:
        df[i].fillna('not_given', inplace=True)
              
    df = df.dropna(axis=0, how='any', subset=['Open', 'StateHoliday', 'SchoolHoliday','CompetitionDistance'])
    return df


def parameter_search(X, y, method, params, random_seed, steps=None):
    random_seed
    valid = {'elastic', 'rf', 'xgboost', 'stack'}
    if method not in valid:
        raise ValueError("Method must be one of %r." % valid)
        
    if method == 'elastic':
        if not type(params) == dict:
            raise ValueError('params is not a dictionary, please support a dictionary of parameters')
        else:
            score = []
            end = np.floor(0.8 * X.shape[0]).astype(int)
            X_train = np.array(X)[:end]
            y_train = np.array(y)[:end]
            X_cv = np.array(X)[end:]
            y_cv = np.array(y)[end:]

        # Grid search for elastic net
        for ratio in params['ratio']:
            for alpha in params['alpha']:
                elastic = ElasticNet(random_state=None, alpha=alpha, l1_ratio=ratio)
                elastic.fit(X_train, y_train)
                preds = elastic.predict(X_cv)                                       
                score.append((alpha, ratio, adam_metric(y_cv, preds)))

                # Get best score
                best_params = min(score, key=itemgetter(2))[0:2]
        
    elif method == 'rf':
        if not type(params) == dict:
            raise ValueError('params is not a dictionary, please support a dictionary of parameters')
        else:            
            score = []
            end = np.floor(0.8 * X.shape[0]).astype(int)
            X_train = np.array(X)[:end]
            y_train = np.array(y)[:end]
            X_cv = np.array(X)[end:]
            y_cv = np.array(y)[end:]

            # Random search for RF
            for step in tqdm(range(steps)):
                n_estimators = np.random.randint(params['n_estimators'][0], params['n_estimators'][1])
                max_depth = np.random.choice(params['max_depth'])
                max_features = np.random.choice(params['max_features'])
                print((n_estimators, max_depth, max_features))

                rf = RandomForestRegressor(n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                max_features=max_features)
                rf.fit(X_train, y_train)
                preds = rf.predict(X_cv)                                       
                score.append((n_estimators, max_depth, max_features, adam_metric(y_cv, preds)))

            # Get best score
            best_params = min(score, key=itemgetter(3))[0:3]
            
    elif method == 'xgboost':
        if not type(params) == dict:
            raise ValueError('params is not a dictionary, please support a dictionary of parameters')
        else:
            score = []
            end = np.floor(0.8 * X.shape[0]).astype(int)
            X_train = np.array(X)[:end]
            y_train = np.array(y)[:end]
            X_cv = np.array(X)[end:]
            y_cv = np.array(y)[end:]

            fit_params = {
                'eval_metric': 'rmse',
                'early_stopping_rounds': 10,    
                'eval_set': [(X_cv, y_cv)]
            }
            # Random search for xgboost
            for step in tqdm(range(steps)):
                n_estimators = int(np.floor(np.random.uniform(params['n_estimators'][0], params['n_estimators'][1])))
                max_depth = np.random.choice(params['max_depth'])
                lr = np.random.choice(params['learning_rate'])
                subsample = np.random.choice(params['subsample'])
                colsample_bytree = np.random.choice(params['colsample_bytree'])
                colsample_bylevel = np.random.choice(params['colsample_bylevel'])
                reg_lambda = np.random.choice(params['reg_lambda'])
                
                print((lr, n_estimators, max_depth, subsample, colsample_bytree, colsample_bylevel, reg_lambda))
                # Train & predict
                xgb_model = xgb.XGBRegressor(learning_rate=lr, 
                                             n_estimators=n_estimators,
                                             max_depth=max_depth,
                                             subsample=subsample,
                                             colsample_bytree=colsample_bytree,
                                             colsample_bylevel=colsample_bylevel,
                                             objective='reg:squarederror')

                xgb_model.fit(X_train, y_train, eval_metric=fit_params['eval_metric'],
                              early_stopping_rounds=fit_params['early_stopping_rounds'], 
                              eval_set=fit_params['eval_set'],
                              verbose=False)
                
                preds = xgb_model.predict(X_cv)                                       
                score.append((lr, n_estimators, max_depth, subsample, colsample_bytree, colsample_bylevel, reg_lambda, \
                              np.sqrt(mean_squared_error(y_cv, preds))))

            # Get best score
            best_params = min(score, key=itemgetter(7))[0:7]
            
    elif method == 'stack':
        pass
    
    return best_params
    
    
    
    