import warnings
import time
import math
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import iqr
import glob
import category_encoders as ce
import skrub
import feature_engine.encoding as fe

sklearn.set_config(transform_output="default")
warnings.filterwarnings("ignore")

"""## Encoding functions definition

### Unsupervised methods
"""

def encode_onehot(df, cat_cols):
    """
    Encodes the specified categorical columns in the DataFrame using one-hot encoding.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be encoded.
    cat_cols (list): A list of column names in df that should be encoded.

    Returns:
    list: A list containing:
        - df_encoded (pd.DataFrame): The DataFrame with the specified categorical columns replaced by their one-hot encoded equivalents.
        - end-start (float): The time taken to perform the encoding, in seconds.
        - dim (float): The ratio of the number of columns in the encoded DataFrame to the number of columns in the original categorical columns.
    """

    encoder = ce.one_hot.OneHotEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_binary(df, cat_cols):

    encoder = ce.binary.BinaryEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_basen(df, cat_cols):

    dim = {}
    max_card = max(df[cat_cols].apply(lambda x: math.ceil(x.nunique())))
    for b in range(1,max_card): #choose a base between 1 and max card with the minimum column number increase
      dim[b]=df[cat_cols].apply(lambda x: math.ceil(x.nunique()/b)).sum()

    min_dim = min(dim.values())
    base = list(dim.keys())[list(dim.values()).index(min_dim)]

    encoder = ce.basen.BaseNEncoder(base = base, return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_count(df, cat_cols):

    encoder = ce.count.CountEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_gray(df, cat_cols):

    encoder = ce.gray.GrayEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_hash(df, cat_cols):

    encoder = ce.hashing.HashingEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_helm(df, cat_cols):

    encoder = ce.helmert.HelmertEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    df_encoded = df_encoded.drop(['intercept'], axis=1) #it adds a variable 'intercept' with only 1 value

    return [df_encoded, end-start, dim]

def encode_ordinal(df, cat_cols):

    encoder = ce.ordinal.OrdinalEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_polynomial(df, cat_cols):

    encoder = ce.polynomial.PolynomialEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    df_encoded = df_encoded.drop(['intercept'], axis=1)

    return [df_encoded, end-start, dim]

def encode_rankhot(df, cat_cols):

    encoder = ce.rankhot.RankHotEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    return [df_encoded, end-start, dim]

def encode_sum(df, cat_cols):

    encoder = ce.sum_coding.SumEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    df_encoded = df_encoded.drop(['intercept'], axis=1)

    return [df_encoded, end-start, dim]

def encode_backdiff(df, cat_cols):

    encoder = ce.backward_difference.BackwardDifferenceEncoder(return_df = True)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    dim = len(encoded_data.columns)/len(df[cat_cols].columns)

    end = time.time()

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_data)

    df_encoded = df_encoded.drop(['intercept'], axis=1)

    return [df_encoded, end-start, dim]

def encode_minhash(df, cat_cols):

    encoder = skrub.MinHashEncoder()

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    end = time.time()

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(cat_cols))

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_df)

    dim = len(encoded_df.columns)/len(df[cat_cols].columns)

    return [df_encoded, end-start, dim]

def encode_gap(df, cat_cols):

    encoder = skrub.GapEncoder(ngram_range=(1, 4), random_state=42)

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    end = time.time()

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(cat_cols))

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_df)

    dim = len(encoded_df.columns)/len(df[cat_cols].columns)

    return [df_encoded, end-start, dim]

def encode_sim(df, cat_cols):

    encoder = skrub.SimilarityEncoder()

    start = time.time()

    encoded_data = encoder.fit_transform(df[cat_cols])

    end = time.time()

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(cat_cols))

    df_encoded = df.drop(cat_cols, axis=1).join(encoded_df)

    dim = len(encoded_df.columns)/len(df[cat_cols].columns)

    return [df_encoded, end-start, dim]

"""### Supervised methods"""

def encode_mean(train, test, cat_cols):
    """
    Encodes the specified categorical columns in the training and testing DataFrames using mean encoding.

    Parameters:
    train (pd.DataFrame): The training DataFrame containing the data to be encoded.
    test (pd.DataFrame): The testing DataFrame containing the data to be encoded.
    cat_cols (list): A list of column names in the DataFrames that should be encoded.

    Returns:
    list: A list containing:
        - encoded_train (pd.DataFrame): The training DataFrame with the specified categorical columns replaced by their mean encoded equivalents.
        - encoded_test (pd.DataFrame): The testing DataFrame with the specified categorical columns replaced by their mean encoded equivalents.
        - end-start (float): The time taken to perform the encoding, in seconds.
        - dim (float): The ratio of the number of columns in the encoded training DataFrame to the number of columns in the original categorical columns.
    """

    encoder = fe.MeanEncoder(variables=cat_cols)

    start = time.time()

    encoded_train = encoder.fit_transform(train, train.target)

    end = time.time()

    dim = len(encoded_train[cat_cols].columns)/len(train[cat_cols].columns)

    encoded_test = encoder.transform(test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_target(train, test, cat_cols):

    encoder = ce.TargetEncoder(cols = cat_cols, return_df = True)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_dt(train, test, cat_cols, task):

    avg = train.target.mean()

    encoder = fe.DecisionTreeEncoder(regression=task, random_state=42, unseen='encode', fill_value=avg)

    start = time.time()

    encoded_train = encoder.fit_transform(train, train.target)

    end = time.time()

    dim = len(encoded_train[cat_cols].columns)/len(train[cat_cols].columns)

    encoded_test = encoder.transform(test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_jamesstein(train, test, cat_cols):

    encoder = ce.james_stein.JamesSteinEncoder(cols = cat_cols, return_df = True, random_state=42)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_catboost(train, test, cat_cols):

    encoder = ce.cat_boost.CatBoostEncoder(cols = cat_cols, return_df = True, random_state=42)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_glmm(train, test, cat_cols):

    encoder = ce.glmm.GLMMEncoder(cols = cat_cols, return_df = True, random_state=42)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_loout(train, test, cat_cols):

    encoder = ce.leave_one_out.LeaveOneOutEncoder(cols = cat_cols, return_df = True, random_state=42)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_mestimate(train, test, cat_cols):

    encoder = ce.m_estimate.MEstimateEncoder(cols = cat_cols, return_df = True, random_state=42)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_quantile(train, test, cat_cols):

    encoder = ce.quantile_encoder.QuantileEncoder(cols = cat_cols, return_df = True)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_summary(train, test, cat_cols):

    encoder = ce.quantile_encoder.SummaryEncoder(cols = cat_cols, return_df = True)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols], override_return_df=True)
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def encode_woe(train, test, cat_cols):

    encoder = ce.woe.WOEEncoder(cols = cat_cols, return_df = True, random_state=42)

    start = time.time()

    encoded_train = encoder.fit_transform(train[cat_cols], train.target)

    end = time.time()

    dim = len(encoded_train.columns)/len(train[cat_cols].columns)

    encoded_train = train.drop(cat_cols, axis=1).join(encoded_train)

    encoded_test = encoder.transform(test[cat_cols])
    encoded_test = test.drop(cat_cols, axis=1).join(encoded_test)

    return [encoded_train, encoded_test, end-start, dim]

def column_types(df):
    """
    Identifies numeric and categorical columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame for which column types are to be identified.

    Returns:
    tuple: A tuple containing:
           - num_cols (list): List of column names identified as numeric.
           - cat_cols (list): List of column names identified as categorical.
    """

    cols = df.columns
    num_cols = df._get_numeric_data().columns.tolist()
    cat_cols = list(set(cols) - set(num_cols))
    if 'target' in cat_cols: cat_cols.remove("target")

    return num_cols, cat_cols

def encode_all(df, cat_cols, tipo):
    """
    Encodes the categorical columns in the DataFrame using various encoding methods.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be encoded.
    cat_cols (list): A list of column names in df that should be encoded.
    tipo (str): The type of problem ('class' for classification, 'regr' for regression).
    seed (int): The seed for random number generation to ensure reproducibility.

    Returns:
    tuple: A tuple containing:
        - dict_df (dict): A dictionary with the encoded DataFrames for each encoding method.
        - times (pd.DataFrame): A DataFrame with the encoding methods, their execution times, and the final dimensions.
    """

    # Factorize target column if necessary
    if tipo == 'class':
      if df['target'].dtype != 'int64':
        df['target'] = pd.factorize(df['target'])[0]

    # Define encoding functions for unsupervised and supervised methods
    unsupervised_encodings = {
        'onehot': encode_onehot,
        'binary': encode_binary,
        'basen': encode_basen,
        'hash': encode_hash,
        'ordinal': encode_ordinal,
        'gray': encode_gray,
        'rankhot': encode_rankhot,
        'count': encode_count,
        'minhash': encode_minhash,
        'similarity': encode_sim,
        'gap': encode_gap,
        'sum': encode_sum,
        'helm': encode_helm,
        'polynomial': encode_polynomial,
        'backdiff': encode_backdiff
    }

    # Sample test and train datasets
    test = df.sample(frac=0.2, random_state=42)
    train = df.drop(test.index)

    supervised_encodings = {
        'mean': encode_mean,
        'target': encode_target,
        'dt': lambda train, test, cat_cols: encode_dt(train, test, cat_cols, task=(tipo == 'regr')),
        'mestimate': encode_mestimate,
        'jamesstein': encode_jamesstein,
        'loout': encode_loout,
        'glmm': encode_glmm,
        'catboost': encode_catboost
    }

    # Initialize dict_df with unsupervised encodings
    dict_df = {name: func(df, cat_cols) for name, func in unsupervised_encodings.items()}

    # Add supervised encodings
    dict_df.update({name: func(train, test, cat_cols) for name, func in supervised_encodings.items()})

    # Conditionally add encodings for regression or binary classification
    unique_targets = df.target.unique()
    if tipo == 'regr':
      dict_df['quantile'] = encode_quantile(train, test, cat_cols)
      dict_df['summary'] = encode_summary(train, test, cat_cols)
      print('target continuo')
    elif len(unique_targets) == 2:
      dict_df['woe'] = encode_woe(train, test, cat_cols)

    # Extract times for each encoding method
    times = pd.DataFrame({'method': list(dict_df.keys())})
    times['time'] = [value[-2] for value in dict_df.values()]
    times['dim_fin'] = [value[-1] for value in dict_df.values()]

    # Remove time information from dict_df
    dict_df = {key: (value[0] if len(value) == 3 else value[:-2]) for key, value in dict_df.items()}
    #dict_df = {k: v[0] if len(v) == 2 else v for k, v in dict_df.items()}

    return (dict_df, times)

"""
### No parameter tuning"""

def predict_it(data, task, target):
    """
    Trains and evaluates a machine-learning models on the given data.

    Parameters:
    data (pd.DataFrame or list of pd.DataFrame): The dataset(s) to be used for training and testing.
    task (bool): The type of machine-learning task; True indicates classification and False regression
    target (str): The name of the target column in the dataset.

    Returns:
    pd.DataFrame: A dataframe containing the model and the relative performance metrics on test set 
    """

    if task:
        models = [
            RandomForestClassifier(random_state=42),
            SVC(random_state=42),
            LogisticRegression(random_state=42), 
            GaussianNB(), 
            MLPClassifier(random_state=42), 
            KNeighborsClassifier(), 
            DecisionTreeClassifier()]
        
        results = pd.DataFrame(columns=['model', 'acc', 'prec', 'recall', 'f1'])
    else:
        models = [
            RandomForestRegressor(random_state=42),
            SVR(random_state=42),
            LinearRegression(random_state=42),  
            MLPRegressor(random_state=42), 
            KNeighborsRegressor(), 
            DecisionTreeRegressor()]
        
        results = pd.DataFrame(columns=['model', 'mae', 'mse', 'rmse', 'nrmse', 'r2'])

    if isinstance(data, list):
        X_train = data[0][[col for col in data[0] if col != target]]
        y_train = data[0][target]

        X_test = data[1][[col for col in data[1] if col != target]]
        y_test = data[1][target]

    else:
        X = data[[col for col in data if col != target]]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test.fillna(y_train.mean(),inplace=True) #fillna only for decision tree encoder, no other dataset has missing values
    
    for model in models:
        clf = Pipeline(
        steps=[('preprocessor',
                StandardScaler()), ('model', model)])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if task:
            res_single = [
                type(model).__name__,
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, zero_division=0, average='macro'),
                recall_score(y_test, y_pred, zero_division=0, average='macro'),
                f1_score(y_test, y_pred, zero_division=0, average='macro')]
        else:
            res_single = [
                type(model).__name__,
                mean_absolute_error(y_test, y_pred),
                mean_squared_error(y_test, y_pred),
                mean_squared_error(y_test, y_pred, squared=False),
                mean_squared_error(y_test, y_pred, squared=False) / iqr(y_test),
                r2_score(y_test, y_pred)]
        results.loc[len(results)] = res_single

    return results

def time_standard(time_df):
    """
    Standardizes time measurements in a DataFrame using Min-Max scaling and classifies the scaled times into quartiles.

    Parameters:
    time_df (pd.DataFrame): The input DataFrame containing a 'time' column with time measurements.

    Returns:
    pd.DataFrame: The DataFrame with two additional columns:
                  - 'time_norm': The Min-Max scaled time measurements.
                  - 'class': The classification of scaled time into four quartiles labeled as 'fast', 'medium-fast', 'medium-slow', 'slow'.
    """

    scaler = MinMaxScaler()
    time_scaled = scaler.fit_transform(time_df['time'].values.reshape(-1, 1))

    time_df.insert(2, 'time_norm', time_scaled)

    time_df.insert(3, 'class', pd.qcut(time_df['time_norm'], 4, labels=["fast", "medium-fast", "medium-slow", "slow"]))

    return time_df

"""## Pipeline finale"""

def final_pip(data, task):
  """
    Executes a pipeline for encoding categorical variables, training multiple models,
    and evaluating their performance and time taken for encoding.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be processed.
    task (bool): The type of machine-learning task; True indicates classification and False regression

    Returns:
    tuple: A tuple containing:
           - res_all (pd.DataFrame): A DataFrame with the performance metrics for each model and encoding method.
           - time_res (pd.DataFrame): A DataFrame with the standardized times for each encoding method along with their final dimensions.
    """
  
  cat_cols = column_types(data)[1]

  print('encoding categorical variables')
  enc = encode_all(data, cat_cols, task) #encoding datasets with different methods
  print('Encoding done')

  enc_dict = enc[0]

  times = enc[1]

  if task:
    res_all = pd.DataFrame(columns=['encoding', 'model', 'acc', 'prec', 'recall', 'f1'])
  else:
    res_all = pd.DataFrame(columns=['encoding', 'model', 'mae', 'mse', 'rmse', 'nrmse', 'r2'])

  for encoder in enc_dict.keys():
    res_encoder = predict_it(enc_dict[encoder], task, 'target')
    res_encoder.insert(0, "encoding", encoder)
    res_all = pd.concat([res_all, res_encoder], ignore_index=True)

  print('Models fitted')

  #time evaluation
  time_res = time_standard(times[['method','time']])
  time_res['dim_fin'] = times['dim_fin']

  return res_all, time_res

#Classification datasets importation

datasets = []

datasets.append(pd.read_csv('../data_clean/class/census.csv'))

datasets.append(pd.read_csv('../data_clean/class/mushrooms.csv'))

churn = pd.read_csv('../data_clean/class/churn.csv')

cols = ['\'number_customer_service_calls\'', '\'state\'', '\'international_plan\'', '\'voice_mail_plan\'', '\'number_customer_service_calls\'']
for col in cols:
    churn[col] = churn[col].astype('str')

datasets.append(churn)

datasets.append(pd.read_csv('../data_clean/class/germancredit.csv'))

datasets.append(pd.read_csv('../data_clean/class/breastcancer.csv'))

datasets.append(pd.read_csv('../data_clean/class/autism_adult.csv'))

datasets.append(pd.read_csv('../data_clean/class/obesity.csv'))

datasets.append(pd.read_csv('../data_clean/class/car.csv'))

cmc = pd.read_csv('../data_clean/class/cmc.csv')
cols = ['wife_edu', 'husband_edu', 'wife_religion',
       'wife_working', 'husband_occupation', 'standard_of_living_index',
       'media_exposure', 'target']
for col in cols:
    cmc[col] = cmc[col].astype('str')

datasets.append(cmc)

datasets.append(pd.read_csv('../data_clean/class/nursery.csv'))

names = ['mush', 'census', 'churn', 'credit', 'breast', 'autism', 'obesity', 'car', 'cmc', 'nursery'] #'census', 'mush', 'churn', 'credit', 'breast', 'autism', 
d = dict(zip(names, datasets))

#Experiment execution for classification datasets

for name in d.keys():
  res = final_pip(d[name], True)
  res[0].to_csv(f'../results/res_class/metrics/results_{name}.csv', index=False)
  res[1].to_csv(f'../results/res_class/time/time_{name}.csv', index=False)
  print(f'{name} done')

#Regression datasets importation

filepaths = glob.glob('/home/clerici/tesi/regr/*.csv')

all_dfs = [pd.read_csv(fp) for fp in filepaths]
names = ['avocado','baseball', 'cmpc2015', 'forest', 'socmob', 'ukair'] 
d = dict(zip(names, all_dfs))

#Experiment execution for regression datasets

for name in list(d.keys()):
  print(name)
  res = final_pip(d[name], False)
  res[0].to_csv(f'../results/res_regr/metrics/results_{name}.csv', index=False)
  res[1].to_csv(f'../results/res_regr/time/time_{name}.csv', index=False)
  print(f'{name} done')