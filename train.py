from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os
import pickle
import pandas as pd
import numpy as np
from mordred import Calculator, descriptors
import xgboost as xgb
import matplotlib.pyplot as plt
# import deepchem as dc

def XGBoost_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dvalid = xgb.DMatrix(X_test, label = y_test)
    dall = xgb.DMatrix(X, label = y)

    # # cross validation for hyperparameters]
    # cv_scores = []
    # for depth in range(2,4):
    #     for rate in range(15,40,5):
    #         rate *= 0.001
    #         # ensemble
    #         va_scores = []
    #         kf = KFold(n_splits=5, shuffle=True, random_state=2)
    #         i = 0
    #         for tr_idx, va_idx in kf.split(X):
    #             i += 1
    #             print(depth,rate,i)
    #             tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
    #             tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]
    #             # plt.hist(tr_idx)
    #             # plt.show()
                
    #             model = xgb.XGBRegressor(max_depth=depth, learning_rate=rate, n_estimators=10000, silent=0, objective="reg:squarederror", random_state=0)
    #             model.fit(tr_x, tr_y, eval_set=[[va_x, va_y]], early_stopping_rounds=100)
    #             va_pred = pd.Series(model.predict(va_x), index=va_idx, name="IC50 (nM)")
    #             va_score = np.mean((va_y-va_pred)**2)
    #             va_scores.append(va_score)

    #         print(sum(va_scores)/5, depth, rate)
    #         cv_scores.append((sum(va_scores)/5,depth,rate))
    
    # print(cv_scores)

    params = {'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 10000, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 0, 'silent': 0}
    num_round = 6000

    # model construction for all
    dfeat = xgb.DMatrix(X, label = y)
    watchlist = [(dfeat, 'feat')]
    model = xgb.train(params, dfeat, num_round, evals = watchlist)
    pickle.dump(model, open(os.path.dirname(__file__) +  "/model_XGB-all.pkl", "wb"))

    # ensemble
    va_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    i = 0
    for tr_idx, va_idx in kf.split(X):
        i += 1
        tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]
        # plt.hist(tr_idx)
        # plt.show()
        
        model = xgb.XGBRegressor(max_depth=3, learning_rate=0.02, n_estimators=10000, silent=0, objective="reg:squarederror", random_state=0)
        model.fit(tr_x, tr_y, eval_set=[[va_x, va_y]], early_stopping_rounds=100)
        va_pred = pd.Series(model.predict(va_x), index=va_idx, name="IC50 (nM)")
        va_score = np.mean((va_y-va_pred)**2)
        print(va_y, va_pred)
        va_scores.append(va_score)
        print(va_score)
 
        pickle.dump(model, open(os.path.dirname(__file__) +  "/model_XGB-" + str(i) + ".pkl", "wb"))

    print(sum(va_scores)/5)


def Ridge_model(X,y):
    # preprocessing
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pickle.dump(scaler, open(os.path.dirname(__file__) +  "/ridge_scaler.pkl", "wb"))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

    model = Ridge(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(np.mean((y_test-y_pred)**2))
    pickle.dump(model, open(os.path.dirname(__file__) +  "/model_Ridge.pkl", "wb"))



def Graph_model():
    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    loader = dc.data.data_loader.CSVLoader(tasks=['IC50 (nM)'], smiles_field='SMILES', id_field='SMILES', featurizer=graph_featurizer)
    dataset = loader.featurize('datasets/dataset.csv')

    splitter = dc.splits.splitters.RandomSplitter()
    trainset, validset = splitter.train_test_split(dataset, frac_train=0.8, seed=0)

    transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=trainset)]

    for transformer in transformers:
        trainset = transformer.transform(trainset)
        validset = transformer.transform(validset)

    graph_model = dc.models.GraphConvModel(n_tasks=1, batch_size=100, graph_conv_layers=[16,16], dense_layer_size=32, dropout=0.1, learning_rate=0.01, mode='regression', model_dir=os.path.dirname(__file__)+"/model_Graph")
    graph_model.fit(trainset, nb_epoch=1000)

    metric = dc.metrics.Metric(dc.metrics.mean_squared_error)

    print("Evaluating model")
    train_scores = graph_model.evaluate(trainset, [metric], transformers)
    valid_scores = graph_model.evaluate(validset, [metric], transformers)

    print("Train scores")
    print(train_scores)

    print("Validation scores")
    print(valid_scores)




if __name__ == "__main__":
    """
    load
    """
    df_train = pd.read_csv("datasets/dataset_last.csv", index_col=0)

    y = df_train["IC50 (nM)"]
    X = df_train.drop(["SMILES", "IC50 (nM)"]
    , axis = 1)
    X_smiles = df_train["SMILES"]

    features = np.load(file=os.path.dirname(__file__)+"/features.npy")
    X = X[features[:500]]


    """
    training
    """
    XGBoost_model(X,y)
    # features = np.load(file=os.path.dirname(__file__)+"/features.npy")
    # Graph_model()
   