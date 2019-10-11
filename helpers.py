import ast
import cufflinks as cf
from datetime import date, datetime
import numpy as np
from operator import itemgetter
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import xgboost as xgb



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

def data_split_transform(df):
    # Split
    end_val = np.floor(0.8 * df.shape[0]).astype(int)
    end_test = np.floor(0.9 * df.shape[0]).astype(int)

    Train = df.loc[:end_val]
    Val = df.loc[end_val:end_test]
    Test = df.loc[end_test:]
    
    Train = data_transformation(Train, type='Train')
    Val = data_transformation(Val,type='Val')
    Test = data_transformation(Test, type='Test')
    
    return Train, Val, Test

def one_hot_enc(Train, Val, Test):
    # OHE
    datalist = [Train, Val, Test]
    cols = Train.select_dtypes(include='object').columns.tolist()
    for data in datalist:
        for col in cols:
            data[col] = data[col].astype(str)
    ohe = OneHotEncoder(handle_unknown='ignore')  

    ohe_train = pd.DataFrame()
    ohe_val = pd.DataFrame()
    ohe_test = pd.DataFrame()

    for col in cols:
        ohe.fit(np.array(Train[col]).reshape(-1,1))
        pickle.dump(ohe, open('metadata/ohe_' + col, "wb"))
        ohe_train_tmp = pd.DataFrame(columns=ohe.categories_, \
                                     data=ohe.transform(np.array(Train[col]).reshape(-1,1)).toarray())
        ohe_val_tmp = pd.DataFrame(columns=ohe.categories_, \
                                     data=ohe.transform(np.array(Val[col]).reshape(-1,1)).toarray()) 
        ohe_test_tmp = pd.DataFrame(columns=ohe.categories_, \
                                     data=ohe.transform(np.array(Test[col]).reshape(-1,1)).toarray()) 
        ohe_train = pd.concat([ohe_train, ohe_train_tmp], axis=1)
        ohe_val = pd.concat([ohe_val, ohe_val_tmp], axis=1)
        ohe_test = pd.concat([ohe_test, ohe_test_tmp], axis=1)
        
    # Drop columns
    for data in datalist:
        data.drop(axis=1, labels=cols, inplace=True)
    
    # Concat
    Train = pd.concat([Train.reset_index(), ohe_train], axis=1)
    Val = pd.concat([Val.reset_index(), ohe_val], axis=1)
    Test = pd.concat([Test.reset_index(), ohe_test], axis=1)
    return Train, Val, Test

def one_hot_enc_test(Test):
    # OHE
    cols = Test.select_dtypes(include='object').columns.tolist()
    for col in cols:
        Test[col] = Test[col].astype(str)
    
    ohe_test = pd.DataFrame()

    for col in cols:
        ohe = pickle.load(open('metadata/ohe_' + col, "rb"))
        ohe_test_tmp = pd.DataFrame(columns=ohe.categories_, \
                                     data=ohe.transform(np.array(Test[col]).reshape(-1,1)).toarray()) 
        ohe_test = pd.concat([ohe_test, ohe_test_tmp], axis=1)
        
    # Drop columns
    Test.drop(axis=1, labels=cols, inplace=True)
    
    # Concat
    Test = pd.concat([Test.reset_index(), ohe_test], axis=1)
    return Test

def data_transformation(df,type='Train'):
    #global global_sales
    # Convert CompetitionYear and CompetitionMonth to datetime format
    df_subset_Comp = df.loc[(~df['CompetitionOpenSinceYear'].isnull()) & (~df['CompetitionOpenSinceMonth'].isnull()), \
                            ['CompetitionOpenSinceYear','CompetitionOpenSinceMonth']]
    df_subset_Comp = float_to_int(df_subset_Comp, {'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'})
    df_subset_Comp['CompetitionStart'] = df_subset_Comp['CompetitionOpenSinceYear'].astype(str) + '-' + \
    df_subset_Comp['CompetitionOpenSinceMonth'].astype(str)  + '-01' 
    df['CompetitionStart'] = pd.to_datetime(df_subset_Comp['CompetitionStart'])
    
    # Calculate is Competition is active and how long the competition is active 
    df['CompetitionActive'] = np.where(df['CompetitionStart'] <= df['Date'], 1, 0)
    df['CompetitionDays'] = (df['Date'] - df['CompetitionStart'])/np.timedelta64(1,'D')
    
    # Convert Promoyear and Promoweekno to datetime format
    df_subset = df.loc[(~df['Promo2SinceYear'].isnull()) & (~df['Promo2SinceWeek'].isnull()), \
                       ['Promo2SinceYear','Promo2SinceWeek']]
    df_subset = float_to_int(df_subset, {'Promo2SinceYear', 'Promo2SinceWeek'})
    df['PromoStart'] = df_subset.apply(lambda row: year_week(row.Promo2SinceYear, row.Promo2SinceWeek), axis=1)

    # create PromoDuration Column:  Date - PromoStart
    df['PromoDuration'] = (df['Date'] - df['PromoStart'])/np.timedelta64(1,'D')
    df['PromoDuration'].fillna(0, inplace=True)
    
    # Create RunnningPromo Column
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
    
    df = df.drop({'Date','CompetitionStart','PromoStart','PromoInterval','Promo','Promo2','DayOfWeek'}, axis=1, errors='ignore') 
    for i in {'Open', 'StateHoliday', 'SchoolHoliday'}:
        df[i].fillna('not_given', inplace=True)
    
    df = df.dropna(axis=0, how='any', subset=['Open', 'StateHoliday', 'SchoolHoliday','CompetitionDistance'])    

    # Nas for comp days
    df['CompetitionDays'].fillna(0, inplace=True)
    
    # set feature COMPETITION INTENSITY
    df['CompetitionIntensity'] = np.log((df['CompetitionDays']*(1/(df['CompetitionDistance'] + 1))+1) + 1e-2)
    df['CompetitionIntensity'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any', subset=['CompetitionIntensity'])    
    
    # calculate mean sales per number of customers per each store type
    df['StoreInfo'] = df['Assortment'] + df['StoreType']
    df['Rel'] = np.nan
    df['ExpectedSales'] = np.nan
    if type == 'Train':
        mean_sales = df.loc[df.Sales > 0, ['Sales', 'Customers', 'StoreInfo']].groupby('StoreInfo').mean()
        mean_sales['Rel'] = mean_sales['Sales']/mean_sales['Customers']
        b = mean_sales['Rel'].to_dict()
        df['Rel'] = df['StoreInfo'].map(b)
        mean_sales['Rel'].to_csv('metadata/MeanSales.csv', header=False)
        df['ExpectedSales'] = df['Customers'] * df['Rel']
        global_sales = np.mean(df['Sales']/df['Customers'])
        with open('global_sales.txt', 'w') as f:
            f.write(str(global_sales))  
    else:
        
        b = pd.read_csv('metadata/MeanSales.csv', header=None, index_col=False).drop(axis=1, labels='index')
        b = b.to_dict()
        for idx, rows in df.iterrows():
            if rows['StoreInfo'] in b.keys():
                rows['Rel'] = b[rows['StoreInfo']]
                rows['ExpectedSales'] = rows['Customers'] * rows['Rel']
            else:
                with open('metadata/global_sales.txt') as f:
                    global_sale = f.read()
                rows['ExpectedSales'] = global_sale
    
    #Set Feature EXPECTED SALES2 (Adam's idea)
    
    
    #print('new Columns created: CompetitionActive, CompetitionDays, PromoDuration, RunningAnyPromo, RunningPromo2, RelativeSales per Number of Customers per StoreType, ExpectedSales')
                
    return df


def parameter_search(X, y, method, params, steps=None):
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
                print(ratio, alpha)
                elastic = ElasticNet(alpha=alpha, l1_ratio=ratio)
                elastic.fit(X_train, y_train)
                preds = elastic.predict(X_cv)                                       
                score.append((alpha, ratio, np.sqrt(mean_squared_error(y_cv, preds))))

                with open('metadata/el_net_params.txt', 'w') as f:
                    f.write(str(score))                    

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
                score.append((n_estimators, max_depth, max_features, np.sqrt(mean_squared_error(y_cv, preds))))
                
                with open('metadata/rf_params.txt', 'w') as f:
                    f.write(str(score))

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
                with open('metadata/xgb_params.txt', 'w') as f:
                    f.write(str(score))

            # Get best score
            best_params = min(score, key=itemgetter(7))[0:7]
            
    elif method == 'stack':
        pass
    
    return best_params

def predict_stacked(X, models, coefs):
    pca = pickle.load(open("metadata/pca.pickle.dat", "rb"))
    X_pca = pca.transform(X)
    valid_models = {'xgb', 'rf', 'el_net'}
    if models.keys() not in valid_models:
        raise ValueError("Model names must be one of %r." % valid_models)
        
    if not type(models) == dict & type(coefs) == dict:
        raise ValueError('models and coefs are dictionaries, please support a dictionary')
    else:    
        xgb_mod = models['xgb']
        rf_mod = models['rf']
        el_net_mod = models['el_net']
        xgb_coef = coefs['xgb']
        rf_coef = coefs['rf']
        el_net_coef = coefs['el_net']
    
    return np.sum(xgb_coef * xgb_mod.predict(X) + rf_coed * rf_mod.predict(X) + el_net_coef * el_net.predict(X_pca))    