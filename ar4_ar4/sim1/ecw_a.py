import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from joblib import Parallel, delayed
import json

N_JOBS = 10
BASE_DIR = Path(__file__).resolve().parent


#########################################################################################################################
# Feature creation and RSS calculator


def RSS_calculator(y, y_hat):
    return np.sum((y - y_hat) ** 2)

def feature_creation(lag_count, df):
    for i in range(1, lag_count + 1):
        df[f'lag_{i}'] = df['label_value'].shift(i).fillna(0)
    return df

#########################################################################################################################
# Loading Hyperparameters for dynamic M2 of A-ECW


def load_m_recent_params(train_exposure):
    params_file = BASE_DIR / "m_recent" / f"train_exposure_{train_exposure}_percent" / "aggregated_hyperparameters.json"
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = json.load(f)
        return {
            'learning_rate': params['learning_rate_median'],
            'num_leaves': int(params['num_leaves_median']),
            'min_data': int(params['min_data_median']),
            'max_depth': int(params['max_depth_median']),
            'boosting_type': params['boosting_type_mode'],
            'n_estimators': int(params['num_boost_round_median'])
        }
    return None

#########################################################################################################################
# Loading BEDM and A-ECW


from batch_error_detection_module import BEDM

def Adaptive_ECW(label_df, lag_count, forecast_h, error_function, train_exposure=None):
    """ 
    Semi Online Adaptive ECW Method with a custom DDM for drift detection 
    
    Contains a stable and a dynamic model retrained by drift detection to ensemble predictions

    -------------------------------------------------------------

    parameters

    label_df : actual observations dataframe column
    lag_count : number of lags to be used as features for retraining dynamic model
    forecast_h : batch length for forecasting, drift detection and drift adaptation
    error_function : function to calculate error for weighting models in ensemble
    train_exposure : to identify the corresponding hyper-parameters
    
    """
    
    weighted_all = pd.DataFrame({
        'df1_predic': label_df.iloc[:, 2].reset_index(drop=True),
        'df2_predic': np.nan,
        'label': label_df.iloc[:, 1].reset_index(drop=True),
        'concept': label_df.iloc[:, 0].reset_index(drop=True),
        'drift_triggered': np.nan,
        'weighted_forecast': np.nan,
        'window_size': pd.Series([None] * len(label_df), dtype="object")    
    })
    window = None
    dynamic_m = None

    # Directly taking the hyperparams for M@
    m_recent_params = load_m_recent_params(train_exposure) 

    for H in range(0, len(weighted_all), forecast_h):
        if H <= 2 * forecast_h:
            for r in range(H, min(H + forecast_h, len(weighted_all))):
                weighted_all.iloc[r, -2] = weighted_all.iloc[r, 0] 
                weighted_all.iloc[r, -1] = 'origin_horizon'
        else:
            window_length = 0
            gd_ddm = BEDM(2, 3)
            gd_ddm.store_concept(
                abs(weighted_all.iloc[H - 2*forecast_h:H - forecast_h, -2] - 
                    weighted_all.iloc[H - 2*forecast_h:H - forecast_h, 2]).mean(),
                abs(weighted_all.iloc[H - 2*forecast_h:H - forecast_h, -2] - 
                    weighted_all.iloc[H - 2*forecast_h:H - forecast_h, 2]).std(),
                list(abs(weighted_all.iloc[H - 2*forecast_h:H - forecast_h, -2] - 
                         weighted_all.iloc[H - 2*forecast_h:H - forecast_h, 2]))
            )
            status = []
            for i in range(H - forecast_h, H):
                status.append(gd_ddm.monitor(
                    abs(weighted_all.iloc[i, -2] - weighted_all.iloc[i, 2]), 2
                ))
            if 'GD detected' in status:
                index = status.index('GD detected')
                window_length = index - 1

            if window_length > 0:
                window = w = window_length
                temp_df = pd.DataFrame({'label_value': weighted_all.iloc[H - forecast_h:H, 2]})
                temp_df = feature_creation(lag_count, temp_df)
                X_train = temp_df.drop(columns=['label_value'])
                y_train = temp_df['label_value']
                
                simple_m = lgb.LGBMRegressor(
                        n_estimators=m_recent_params['n_estimators'],
                        max_depth=m_recent_params['max_depth'],
                        learning_rate=m_recent_params['learning_rate'],
                        num_leaves=m_recent_params['num_leaves'],
                        min_child_samples=m_recent_params['min_data'],
                        boosting_type=m_recent_params['boosting_type'],
                        objective='regression',
                        device='cpu',
                        num_threads=1,
                        random_state=42,
                        verbose=-1 
                    )
                
                simple_m.fit(X_train, y_train)
                dynamic_m = simple_m

                for r in range(H, min(H + forecast_h, len(weighted_all))):
                    lags = [f"lag_{i}" for i in range(1, lag_count + 1)]
                    test_pred_X = pd.DataFrame([weighted_all.iloc[r - lag_count:r, 2].values], columns=lags)
                    pred = simple_m.predict(test_pred_X)
                    weighted_all.iloc[r, 1] = pred[0]
                    weighted_all.iloc[r, -3] = 1
                    if r < w:
                        weighted_all.iloc[r, -2] = 0.5 * weighted_all.iloc[r, 0] + 0.5 * weighted_all.iloc[r, 1]
                        weighted_all.iloc[r, -1] = f'size_{w}'
                    else:
                        m1_error = error_function(weighted_all.iloc[r - w:r, 0],
                                                weighted_all.iloc[r - w:r, 2])
                        m2_error = error_function(weighted_all.iloc[r - w:r, 1],
                                                weighted_all.iloc[r - w:r, 2])
                        error = m1_error + m2_error
                        w1 = 0.5 if error == 0 else m2_error / error
                        weighted_all.iloc[r, -2] = w1 * weighted_all.iloc[r, 0] + (1 - w1) * weighted_all.iloc[r, 1]
                        weighted_all.iloc[r, -1] = f'size_{w}'
            else:
                w = window if window is not None else 1
                for r in range(H, min(H + forecast_h, len(weighted_all))):
                    if dynamic_m is None:                    
                        weighted_all.iloc[r, -2] = weighted_all.iloc[r, 0]
                        weighted_all.iloc[r, -1] = f'size not detected'
                    else:
                        lags = [f"lag_{i}" for i in range(1, lag_count + 1)]
                        test_pred_X = pd.DataFrame([weighted_all.iloc[r - lag_count:r, 2].values], columns=lags)
                        pred = dynamic_m.predict(test_pred_X)
                        weighted_all.iloc[r, 1] = pred[0]
                        weighted_all.iloc[r, -3] = 1
                        if r < w:
                            weighted_all.iloc[r, -2] = 0.5 * weighted_all.iloc[r, 0] + 0.5 * weighted_all.iloc[r, 1]
                            weighted_all.iloc[r, -1] = f'size_{w}'
                        else:
                            m1_error = error_function(weighted_all.iloc[r - w:r, 0],
                                                    weighted_all.iloc[r - w:r, 2])
                            m2_error = error_function(weighted_all.iloc[r - w:r, 1],
                                                    weighted_all.iloc[r - w:r, 2])
                            error = m1_error + m2_error
                            w1 = 0.5 if error == 0 else m2_error / error
                            weighted_all.iloc[r, -2] = w1 * weighted_all.iloc[r, 0] + (1 - w1) * weighted_all.iloc[r, 1]
                            weighted_all.iloc[r, -1] = f'size_{w}'

    return weighted_all

#########################################################################################################################


def run_adaptive_ecw_for_path(path, te, e, model_history, all_test_dict, output_dir):
    """Runs Adaptive_ECW for ONE path and saves result."""
    
    df = Adaptive_ECW(
        label_df=all_test_dict[model_history[0]]
                                [f"train_exposure_{e}"]
                                [f"test_exposure_{te}_percent"]
                                [path],
        lag_count=10,
        forecast_h=300,
        error_function=RSS_calculator,
        train_exposure=e
    )

    save_dir = output_dir / f"{te}" / f"{e}"
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / f"{path}.csv", index=False)

    del df


#########################################################################################################################

if __name__ == "__main__":
    all_test_dict = {}
    train_exposures = [0,5,10,20,30]
    test_exposures = [40,50,60]
    model_history = ["m_all"]
    test_paths = []

    for model in model_history:
        model_pred = {}
        for e in train_exposures:
            train_exposure_dict = {}
            for te in test_exposures:
                folder = (BASE_DIR / model / f"train_exposure_{e}_percent" / 
                         f"test_exposure_{te}_percent")
                
                temp_dict = {}
                for file in folder.glob("*.csv"):
                    path_number = file.stem.split("_")[-1]
                    path_name = f"path_{path_number}"
                    temp_dict[path_name] = pd.read_csv(file)
                    if path_name not in test_paths:
                        test_paths.append(path_name)
                
                train_exposure_dict[f'test_exposure_{te}_percent'] = temp_dict
            model_pred[f"train_exposure_{e}"] = train_exposure_dict
        all_test_dict[f"{model}"] = model_pred

    output_dir = BASE_DIR / "a_ecw"
    output_dir.mkdir(parents=True, exist_ok=True)

    for te in test_exposures:
        for e in train_exposures:
            print(f"\nProcessing test_exposure={te}, train_exposure={e}")
            Parallel(
                n_jobs=N_JOBS,
                backend="loky"
            )(
                delayed(run_adaptive_ecw_for_path)(
                    path, te, e, model_history, all_test_dict, output_dir
                )
                for path in test_paths
            )