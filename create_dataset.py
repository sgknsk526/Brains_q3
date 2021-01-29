from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from pandas import DataFrame
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import BulkTanimotoSimilarity
from mordred import Calculator, descriptors
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import pickle

def rdkit_features(mols):
    descs = dict()

    for desc in Descriptors.descList:
        feat = 'rd_'+str(desc[0])
        print(feat)
        descs[feat] = []
        for mol in mols:
            try:
                descs[feat].append(desc[1](mol))
            except:
                descs[feat].append(None)

    desc_df = pd.DataFrame(descs)
    return desc_df

def mordred_features(mols):
    calc = Calculator(descriptors, ignore_3D=True)
    df_descriptors = calc.pandas(mols, nproc=4)
    df_descriptors = df_descriptors.astype(str)
    masks = df_descriptors.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
    df_descriptors = df_descriptors[~masks]
    df_descriptors = df_descriptors.astype(float)
    df_descriptors.head()
    return df_descriptors

def morgan_fingerprints(mols):
    fing_maps = [GetMorganFingerprint(mol, 2).GetNonzeroElements() for mol in mols]

    morgan_keys = set()
    for maps in fing_maps:
        for k in maps.keys():
            morgan_keys.add(k)
    morgan_keys = list(morgan_keys)
    
    morgan_mat = np.zeros((len(fing_maps), len(morgan_keys)))
    for idx, maps in enumerate(fing_maps):
        for k, v in maps.items():
            morgan_mat[idx][morgan_keys.index(k)] = v

    morgan_fing = pd.DataFrame(morgan_mat, columns=morgan_keys)

    return morgan_fing

def simiralities(mols):
    ref_fps = np.load(file=os.path.dirname(__file__)+"/ref_fps.npy", allow_pickle=True)
    mol_fps = [FingerprintMols.FingerprintMol(mol) for mol in mols]

    d = dict()
    col_names = []
    for i,r_fp in enumerate(ref_fps):
        col_name = "SM_"+str(i)
        d[col_name] = BulkTanimotoSimilarity(r_fp, mol_fps)
        print(i)
        col_names.append(col_name)
    
    sims = pd.DataFrame(d, columns=col_names)

    return sims

if __name__ == "__main__":
    """
    rdkit+morgan_fing+mordred descriptors
    """
    # df = pd.read_csv("datasets/dataset_original.csv")
    # mols = df["SMILES"].apply(Chem.MolFromSmiles)
    # X = rdkit_features(mols)
    # output = pd.concat([df, X], axis=1)
    # X = mordred_features(mols)
    # output = pd.concat([output, X], axis=1)
    # X = morgan_fingerprints(mols)
    # output = pd.concat([output, X], axis=1)
    # output.to_csv('datasets/dataset_all.csv')
    # """
    # X = morgan_fingerprints(mols)
    # output = pd.concat([output, X], axis=1)
    # output.to_csv('datasets/data_processed.csv')
    # """
    # print(output)

    # # cleaning
    # df_train = pd.read_csv("datasets/dataset_all.csv", index_col=0)
    # print(df_train.head())

    # # model construction
    # y = DataFrame(np.log1p(df_train["IC50 (nM)"]))
    # X = df_train.drop(["SMILES", "IC50 (nM)"], axis = 1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # dtrain = xgb.DMatrix(X_train, label = y_train)
    # dvalid = xgb.DMatrix(X_test, label = y_test)
    # dall = xgb.DMatrix(X, label = y)
    # params = {'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 20000, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 0, 'silent': 0}
    # num_round = 6000
    # watchlist = [(dall, 'all')]
    # model = xgb.train(params, dall, num_round, evals = watchlist)
    # pickle.dump(model, open("model_for_dataset/model_inst.pkl", "wb"))
    
    # main_features = []
    # for k,v in model.get_score(importance_type='total_gain').items():
    #     main_features.append(k)

    # X_main = X[main_features]
    # print(X_main)
    
    """
    squared descriptors
    """
    # X_j2 = dict()
    # for feat in X.columns:
    #     X_j2['2j-'+feat] = X[feat]**2
    # X_j2 = DataFrame(X_j2)

    # X_train, X_test, y_train, y_test = train_test_split(X_j2, y, test_size = 0.2, random_state = 0)
    # dtrain = xgb.DMatrix(X_train, label = y_train)
    # dvalid = xgb.DMatrix(X_test, label = y_test)
    # dall = xgb.DMatrix(X_j2, label = y)
    # params = {'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 20000, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 0, 'silent': 0}
    # num_round = 6000
    # watchlist = [(dall, 'all')]
    # model = xgb.train(params, dall, num_round, evals = watchlist)
    # pickle.dump(model, open("model_for_dataset/model_inst2.pkl", "wb"))

    # main_features = []
    # for k,v in model.get_score(importance_type='total_gain').items():
    #     main_features.append(k)

    # X_j2_main = X_j2[main_features]
    # print(X_j2_main)

    # df = pd.concat([df_train['SMILES'],y], axis=1)
    # df = pd.concat([df,X_main], axis=1)
    # df = pd.concat([df,X_j2_main], axis=1)
    # df.to_csv('datasets/dataset_+j2.csv')
    # print(df)
    
    """
    similarity descriptors
    """
    # df = pd.read_csv("datasets/dataset_+j2.csv", index_col=0)

    # mols = df["SMILES"].apply(Chem.MolFromSmiles)
    # refs = [FingerprintMols.FingerprintMol(x) for x in mols]
    # np.save(os.path.dirname(__file__) + "/ref_fps", refs)

    # sims = simiralities(mols)
    # df = pd.concat([df,sims], axis=1)
    # df.to_csv('datasets/dataset_sims.csv')

    """
    choose useful descriptors
    """
    df = pd.read_csv("datasets/dataset_sims.csv", index_col=0)
    print(df)

    y = df["IC50 (nM)"]
    X = df.drop(["SMILES", "IC50 (nM)"], axis=1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # dtrain = xgb.DMatrix(X_train, label = y_train)
    # dvalid = xgb.DMatrix(X_test, label = y_test)
    # dall = xgb.DMatrix(X, label = y)
    # params = {'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 10000, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': 0, 'silent': 0}
    # num_round = 6000
    # watchlist = [(dall, 'all')]
    # model = xgb.train(params, dall, num_round, evals = watchlist)
    # pickle.dump(model, open("model_for_dataset/model_selected.pkl", "wb"))
    
    model = pickle.load(open("model_for_dataset/model_selected.pkl", "rb"))
    features_list = []
    for k,v in model.get_score(importance_type='total_gain').items():
        features_list.append((v,k))
    
    features_list.sort(reverse=True)

    main_features = []
    for i in features_list:
        main_features.append(i[1])

    rd_features = []
    md_features = []
    mg_features = []
    sm_features = []

    non_features = set()
    i = 0

    for feat in main_features:
        feat_was = feat
        if feat[:3] == '2j-':
            feat = feat[3:]
        if feat[:3] == 'SM_':
            sm_features.append(feat[3:])
        elif feat[:3] == 'rd_':
            rd_features.append(feat)
        elif all(fi.isdigit() for fi in feat):
            mg_features.append(feat)
        else:
            if i < 100:
                md_features.append(feat)
                i += 1
            else:
                non_features.add(feat_was)
    
    features = []
    for feat in main_features:
        if not feat in non_features:
            features.append(feat)

        
    
    print(len(rd_features), len(md_features), len(mg_features), len(sm_features))
    
    rd_features = list(set(rd_features))
    md_features = list(set(md_features))
    mg_features = list(set(mg_features))

    np.save(os.path.dirname(__file__) + "/rd_features", rd_features)
    np.save(os.path.dirname(__file__) + "/md_features", md_features)
    np.save(os.path.dirname(__file__) + "/mg_features", mg_features)
    np.save(os.path.dirname(__file__) + "/sm_features", sm_features)
    np.save(os.path.dirname(__file__) + "/features", features)

    X_main = X[features]
    df = pd.concat([df['SMILES'],y], axis=1)
    df = pd.concat([df,X_main], axis=1)
    df.to_csv('datasets/dataset_last.csv')

    mols = df["SMILES"].apply(Chem.MolFromSmiles)
    refs = [FingerprintMols.FingerprintMol(x) for x in mols]
    ref_fps = []
    for i in sm_features:
        ref_fps.append(refs[int(i)])
    np.save(os.path.dirname(__file__) + "/ref_fps", ref_fps)

    print(sm_features,len(sm_features))
    print(ref_fps,len(ref_fps))
    print(df)

