import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_train = pd.read_csv('train.csv');
data_test = pd.read_csv('test.csv');

# ,,,,,,,,
data_train.loc[:, ('NU_NOTA_CN')] = np.nan_to_num(data_train['NU_NOTA_CN']);
data_train.loc[:, ('NU_NOTA_CH')] = np.nan_to_num(data_train['NU_NOTA_CH']);
data_train.loc[:, ('NU_NOTA_LC')] = np.nan_to_num(data_train['NU_NOTA_LC']);
data_train.loc[:, ('NU_NOTA_REDACAO')] = np.nan_to_num(data_train['NU_NOTA_REDACAO']);
data_train.loc[:, ('NU_NOTA_MT')] = np.nan_to_num(data_train['NU_NOTA_MT']);
data_test.loc[:, ('NU_NOTA_CN')] = np.nan_to_num(data_test['NU_NOTA_CN']);
data_test.loc[:, ('NU_NOTA_CH')] = np.nan_to_num(data_test['NU_NOTA_CH']);
data_test.loc[:, ('NU_NOTA_LC')] = np.nan_to_num(data_test['NU_NOTA_LC']);
data_test.loc[:, ('NU_NOTA_REDACAO')] = np.nan_to_num(data_test['NU_NOTA_REDACAO']);

data_train = data_train[['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO','NU_NOTA_MT']];
data_test = data_test[['NU_INSCRICAO','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']];

polynomialFeatures = PolynomialFeatures(degree = 8)
XTrain = polynomialFeatures.fit_transform(np.array(data_train.loc[:,('NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO')]));
XTest = polynomialFeatures.fit_transform(np.array(data_test.loc[:,('NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO')]));

regressor = LinearRegression();

regressor.fit(XTrain, data_train.loc[:,('NU_NOTA_MT')]);
YPred = regressor.predict(XTest);

alunos2 = pd.DataFrame();
alunos2.loc[:,('NU_INSCRICAO')] = data_test.loc[:,('NU_INSCRICAO')];
alunos2.loc[:,('NU_NOTA_MT')] = YPred;
alunos2.loc[:,('NU_INSCRICAO','NU_NOTA_MT')].to_csv('answer.csv',index=False);