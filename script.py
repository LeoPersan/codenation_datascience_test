import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_train = pd.read_csv('train.csv');
data_test = pd.read_csv('test.csv');

data_train.loc[:, ('NU_NOTA_CN')] = np.nan_to_num(data_train['NU_NOTA_CN']);
data_train.loc[:, ('NU_NOTA_CH')] = np.nan_to_num(data_train['NU_NOTA_CH']);
data_train.loc[:, ('NU_NOTA_LC')] = np.nan_to_num(data_train['NU_NOTA_LC']);
data_train.loc[:, ('NU_NOTA_REDACAO')] = np.nan_to_num(data_train['NU_NOTA_REDACAO']);
data_train.loc[:, ('NU_NOTA_MT')] = np.nan_to_num(data_train['NU_NOTA_MT']);
data_test.loc[:, ('NU_NOTA_CN')] = np.nan_to_num(data_test['NU_NOTA_CN']);
data_test.loc[:, ('NU_NOTA_CH')] = np.nan_to_num(data_test['NU_NOTA_CH']);
data_test.loc[:, ('NU_NOTA_LC')] = np.nan_to_num(data_test['NU_NOTA_LC']);
data_test.loc[:, ('NU_NOTA_REDACAO')] = np.nan_to_num(data_test['NU_NOTA_REDACAO']);

data_train = data_train[['TP_SEXO','TP_ESCOLA','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO','NU_NOTA_MT']];
data_test = data_test[['NU_INSCRICAO','TP_SEXO','TP_ESCOLA','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']];

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

for column in ['TP_SEXO']:
    data_train.loc[:, (column)] = labelencoder_X.fit_transform(data_train.loc[:, (column)]);
    D = pd.get_dummies(data_train.loc[:,(column)]).values;
    if D.shape[1] > 1:
        D = D[:,1:];
    data_train = data_train.drop(columns=column);
    for j in range(0, D.shape[1]):
        data_train.insert(len(data_train.columns)-1,column+'_'+str(j),D[:,j]);

    data_test.loc[:, (column)] = labelencoder_X.fit_transform(data_test.loc[:, (column)]);
    D = pd.get_dummies(data_test.loc[:,(column)]).values;
    if D.shape[1] > 1:
        D = D[:,1:];
    data_test = data_test.drop(columns=column);
    for j in range(0, D.shape[1]):
        data_test.insert(len(data_test.columns),column+'_'+str(j),D[:,j]);

polynomialFeatures = PolynomialFeatures(degree=8, interaction_only=True, include_bias=False)
XTrain = polynomialFeatures.fit_transform(np.array(data_train.iloc[:,:-1]));
XTest = polynomialFeatures.fit_transform(np.array(data_test.iloc[:,1:]));

import statsmodels.api as sm
XTrain = np.insert(XTrain, 0, 1, axis=1)
XTest = np.insert(XTest, 0, 1, axis=1)
numVars = len(XTrain[0])
for i in range(0, numVars):
    regressor = sm.OLS(data_train.loc[:,('NU_NOTA_MT')], XTrain.astype(float)).fit();
    maxVar = max(regressor.pvalues);
    if maxVar > 0.05:
        for j in range(0, len(regressor.pvalues)):
            if (regressor.pvalues[j].astype(float) == maxVar):
                XTrain = np.delete(XTrain, j, 1);
                XTest = np.delete(XTest, j, 1);
XTrain = XTrain[:,1:];
XTest = XTest[:,1:];

regressor = LinearRegression();

regressor.fit(XTrain, data_train.loc[:,('NU_NOTA_MT')]);
YPred = regressor.predict(XTest);

alunos2 = pd.DataFrame();
alunos2.loc[:,('NU_INSCRICAO')] = data_test.loc[:,('NU_INSCRICAO')];
alunos2.loc[:,('NU_NOTA_MT')] = YPred;
alunos2.loc[:,('NU_INSCRICAO','NU_NOTA_MT')].to_csv('answer.csv',index=False);