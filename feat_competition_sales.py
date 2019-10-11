from helpers import *
import statsmodels.api as sm
import statsmodels.formula.api as smf

stores = pd.read_csv("data/store.csv")
train = pd.read_csv("data/train.csv")
train.head()

# must get dates right before running the other functions
train = date_convert(train)

# merge stores into train
full_train = pd.merge(train, stores, how="left", on="Store")
full_train.head()

def create_features_competition(full_train):
    # convert NAs to 0 so the conversion to int is possible
    full_train["CompetitionOpenSinceYear"] = full_train["CompetitionOpenSinceYear"].fillna(1800)
    full_train["CompetitionOpenSinceMonth"] = full_train["CompetitionOpenSinceMonth"].fillna(1)
    # convert month/year to int
    full_train["CompetitionOpenSinceMonth"] = full_train["CompetitionOpenSinceMonth"].astype(int).astype(str)
    full_train["CompetitionOpenSinceYear"] = full_train["CompetitionOpenSinceYear"].astype(int).astype(str)
    full_train['competition_started'] = full_train['CompetitionOpenSinceYear'] + '-' + full_train['CompetitionOpenSinceMonth']  + '-01' 
    full_train['competition_started'] = pd.to_datetime(full_train["competition_started"])
    # set all dates prior 1900 to Nan
    full_train.loc[full_train.competition_started < '1900-01-01', "competition_started" ] = np.nan
    full_train["competition_active"] = np.where(full_train["competition_started"] <= full_train["Date"], 1, 0)
    full_train.loc[pd.isnull(full_train["competition_started"]) == True, "competition_active"] = np.nan
    # set feature COMPETITION DAYA
    # how long has the competition been active?
    full_train['competition_days'] = full_train['Date'] - full_train['competition_started']
    full_train.loc[full_train['competition_days'].dt.days < 0, 'competition_days'] = np.nan
    full_train['competition_days'] = full_train['competition_days'].dt.days
    # set feature COMPETITION INTENSITY
    full_train['competition_intensity'] = np.log(full_train['competition_days']*(1/(full_train['CompetitionDistance']))+1)
    
    return full_train

def create_features_customers(full_train):
    full_train['store_info'] = full_train['Assortment'] + full_train['StoreType']
    '''
        #fit OLS
        mod = smf.ols(formula='Sales ~ Customers + store_info -1', data=full_train)
        res = mod.fit()
        customer_weight = res.params[9]
        storetype_weights = res.params[0:9]
        storetype_weights.index = ['aa','ab', 'ac', 'ad', 'bb', 'ca', 'cb', 'cc', 'cd']
        full_train['expected_sales'] = np.nan
        full_train['weight'] = np.nan
        a = storetype_weights.to_dict()
        full_train['weight'] = full_train['store_info'].map(a)
        #Set Feature EXPECTED SALES
        full_train.expected_sales[full_train.Sales > 0] = (full_train.Customers[full_train.Sales > 0] * customer_weight) +\
        full_train.weight[full_train.Sales > 0]
    '''   
    # calculate mean sales per number of customers per each store type
    mean_sales = full_train.loc[full_train.Sales > 0, ['Sales', 'Customers', 'store_info']].groupby('store_info').mean()
    mean_sales['rel'] = mean_sales['Sales']/mean_sales['Customers']
    b = mean_sales['rel'].to_dict()
    full_train['rel'] = full_train['store_info'].map(b)
    
    #Set Feature EXPECTED SALES2 (Adam's idea)
    full_train['expected_sales'] = np.nan
    full_train['expected_sales'] = full_train['Customers'] * full_train['rel']

    return full_train

# set list of features that will be used for XGBoost
f_features =  ['Year', 'Quarter', 'Month', 'Week', 'Day', 'Date', 'Store', 'DayOfWeek',
       'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
       'StoreType', 'Assortment', 'CompetitionDistance',
       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
       'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 
       'competition_active', 'competition_days', 'competition_intensity',
       'expected_sales', 'rel'] #'expected_sales2']