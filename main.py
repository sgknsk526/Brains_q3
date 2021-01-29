from rdkit import Chem
from rdkit.Chem import Descriptors
import sys
import os
import pickle
import pandas as pd
import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem.AllChem import GetMorganFingerprint
import warnings
import xgboost as xgb
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import BulkTanimotoSimilarity
import time


def rdkit_features(mols,rd_features):
    descs = dict()
    st = set(rd_features)

    for feat in rd_features:
        descs[feat] = []

    for desc in Descriptors.descList:
        feat = 'rd_'+str(desc[0])
        if feat in st:
            for mol in mols:
                try:
                    descs[feat].append(desc[1](mol))
                except:
                    descs[feat].append(np.nan)

    desc_df = pd.DataFrame(descs)
    return desc_df

def mordred_features(mols,md_features):
    calc_dummy = Calculator(descriptors, ignore_3D=True)
    st = set(md_features)
    my_descs = []
    for desc in calc_dummy.descriptors:
        if str(desc) in st:
            my_descs.append(desc)

    calc_real = Calculator(my_descs, ignore_3D=True)
    df_descriptors = calc_real.pandas(mols, nproc=4, quiet=True)
    df_descriptors = df_descriptors.astype(str)
    masks = df_descriptors.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
    df_descriptors = df_descriptors[~masks]
    df_descriptors = df_descriptors.astype(float)

    return df_descriptors

def morgan_fingerprints(mols,mg_features):
    fing_maps = [GetMorganFingerprint(mol, 2).GetNonzeroElements() for mol in mols]
    st = set(mg_features)
    
    morgan_mat = np.zeros((len(fing_maps), len(mg_features)))
    for idx, maps in enumerate(fing_maps):
        for i, feat in enumerate(mg_features):
            if int(feat) in maps.keys():    
                morgan_mat[idx][i] = maps[int(feat)]

    morgan_fing = pd.DataFrame(morgan_mat, columns=mg_features)

    return morgan_fing

def simiralities(mols,sm_features):
    ref_fps = np.load(file=os.path.dirname(__file__)+"/ref_fps.npy", allow_pickle=True)
    mol_fps = [FingerprintMols.FingerprintMol(mol) for mol in mols]

    d = dict()
    col_names = []
    for i,r_fp in zip(sm_features,ref_fps):
        col_name = "SM_"+str(i)
        d[col_name] = BulkTanimotoSimilarity(r_fp, mol_fps)
        col_names.append(col_name)
    
    sims = pd.DataFrame(d, columns=col_names)

    return sims

def XGBoost_prediction(X_test):
    # prediction
    preds = []
    for i in range(5):
        model = pickle.load(open(os.path.dirname(__file__) + "/model_XGB-" + str(i+1) + ".pkl", "rb"))
        y_pred = model.predict(X_test)
        preds.append(y_pred)
    
    # mean
    scores = []
    for j in range(len(preds[0])):
        score = 0.0 
        for i in range(5):
            score += preds[i][j]
        score /= 5
        scores.append(score)
    
    # # big model
    # X_test = xgb.DMatrix(X_test)
    # model = pickle.load(open(os.path.dirname(__file__) + "/model_XGB-all.pkl", "rb"))
    # all_scores = model.predict(X_test)

    return scores

def Ridge_prediction(X_test):
    # preprocessing
    X_test = X_test.fillna(0)
    scaler = pickle.load(open(os.path.dirname(__file__) + "/ridge_scaler.pkl", "rb"))
    X_test = scaler.transform(X_test)

    model = pickle.load(open(os.path.dirname(__file__) + "/model_Ridge.pkl", "rb"))
    y_pred = model.predict(X_test)

    return y_pred


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    t = time.time()
    smiles = []
    for line in sys.stdin:
        smiles.append(line.strip())


    features = np.load(file=os.path.dirname(__file__)+"/features.npy")
    rd_features = np.load(file=os.path.dirname(__file__)+"/rd_features.npy")
    md_features = np.load(file=os.path.dirname(__file__)+"/md_features.npy")
    mg_features = np.load(file=os.path.dirname(__file__)+"/mg_features.npy")
    sm_features = np.load(file=os.path.dirname(__file__)+"/sm_features.npy")


    smiles = pd.DataFrame(data=smiles, columns=['SMILES'])

    mols = smiles["SMILES"].apply(Chem.MolFromSmiles)
    df_test = rdkit_features(mols,rd_features)
    X = mordred_features(mols,md_features)
    df_test = pd.concat([df_test,X], axis=1)
    X = morgan_fingerprints(mols,mg_features)
    df_test = pd.concat([df_test,X], axis=1)

    # print(time.time()-t)

    j2_cols = []
    j2_name = []
    for feat in features:
        if feat[:3] == '2j-':
            j2_name.append(feat)
            j2_cols.append(feat[3:])

    # print(df_test)
    df_j2 = df_test[j2_cols]**2
    df_j2.columns = j2_name


    df_test = pd.concat([df_test,df_j2], axis=1)
    X = simiralities(mols, sm_features)
    df_test = pd.concat([df_test,X], axis=1)
    df_test = df_test[features[:500]]


    X_scores = XGBoost_prediction(df_test)

    # R_scores = Ridge_prediction(df_test)
    for Xi in X_scores:
        print(np.expm1(Xi))
    