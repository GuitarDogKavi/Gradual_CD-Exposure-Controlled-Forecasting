import pandas as pd
import itertools
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import json
from scipy import stats
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from joblib import Parallel, delayed

N_JOBS = 8
BASE_DIR = Path(__file__).resolve().parent
File_DIR = BASE_DIR.parent

#########################################################################################################################
# Train Data Loading


train_dict = {}
train_exposures = [0,5,10,20,30]

training_data_dir = File_DIR / "training_data"

all_files = list(training_data_dir.glob("cts_path_*_exposure_*.csv"))

path_numbers = sorted(set(
    int(f.stem.split("_")[2])
    for f in all_files
))

selected_path_number = path_numbers[0]
print(f"Selected path number: {selected_path_number}")

for i in train_exposures:
    file_path = training_data_dir / f"cts_path_{selected_path_number}_exposure_{i}%.csv"
    train_dict[f"train_exposure_{i}_percent"] = pd.read_csv(file_path)

#########################################################################################################################


def para_num(num_lr,num_sub_feature,Num_leaves,num_min_data,num_max_depth):
    LR_list=[np.random.uniform(0, 1) for i in range(num_lr) ]
    LR_list.sort()

    sub_feature_list=[np.random.uniform(0, 1) for i in range(num_sub_feature)]
    sub_feature_list.sort()

    num_leaves_list=[np.random.randint(20, 300) for i in range(Num_leaves)]
    num_leaves_list.sort()

    min_data_list=[np.random.randint(10, 100) for i in range(num_min_data)]
    min_data_list.sort()

    max_depth_list=[np.random.randint(50, 300) for i in range(num_max_depth)]
    max_depth_list.sort()
    boost_type_list=['goss']
    return LR_list,sub_feature_list,num_leaves_list,min_data_list,max_depth_list,boost_type_list

def my_prequential_CV(K, training_set, LR_list, sub_feature_list, num_leaves_list, min_data_list, max_depth_list, boost_type_list):
    ave_training_RMSE = 0
    ave_training_MAE = float('inf')
    for learning_rate in LR_list:
        for boosting_type in boost_type_list:
            for sub_feature in sub_feature_list:
                for num_leaves in num_leaves_list:
                    for min_data in min_data_list:
                        for max_depth in max_depth_list:
                            params = {'learning_rate': learning_rate,
                                      'boosting_type': boosting_type,
                                      'sub_feature': sub_feature,
                                      'num_leaves': num_leaves,
                                      'min_data': min_data,
                                      'max_depth': max_depth,
                                      'verbosity': -1,
                                      'feature_pre_filter': False}

                            valid_pre_list = list()
                            rows = training_set.count()[0]
                            sum_metric = 0
                            for k in range(K):
                                if k == K - 1:
                                    break
                                train = training_set.iloc[:round((k + 1) * rows / K)]
                                valid = training_set.iloc[round((k + 1) * rows / K):round((k + 2) * rows / K)]
                                train_x = train.drop(columns=['value'], axis=1)
                                train_y = train['value']

                                valid_lable = valid['value']
                                valid_x = valid.drop(columns=['value'], axis=1)

                                train_set = lgb.Dataset(train_x, label=train_y)

                                model_gbm = lgb.train(params,
                                                      train_set)

                                valid_pre = model_gbm.predict(valid_x)
                                valid_pre_list.append(valid_pre)

                            all_predic = np.array(list(itertools.chain.from_iterable(valid_pre_list)))
                            valid_label = training_set.iloc[round((1) * rows / K):]['value']
                            RMSE = np.sqrt(((all_predic - valid_label) ** 2).mean())
                            MAE = mean_absolute_error(all_predic, valid_label)

                            if MAE < ave_training_MAE:
                                best_params_dic = {
                                    'learning_rate': learning_rate,
                                    'boosting_type': boosting_type,
                                    'sub_feature': sub_feature,
                                    'num_leaves': num_leaves,
                                    'min_data': min_data,
                                    'max_depth': max_depth}
                                ave_training_RMSE = RMSE
                                ave_training_MAE = MAE

    return best_params_dic

def feature_creation(df, column_index, lag_count = 10):
    for i in range(1,lag_count+1,1):
        df[f'lag_{i}'] = df.iloc[:,column_index].shift(i).fillna(0)
    return df

def aggregate_hyperparameters(params_list):
    if not params_list:
        return {}
    
    aggregated = {}
    param_names = params_list[0].keys()
    
    for param in param_names:
        values = [p[param] for p in params_list]
        
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            aggregated[f'{param}_mean'] = float(np.mean(values))
            aggregated[f'{param}_median'] = float(np.median(values))
        else:
            unique_vals, counts = np.unique(values, return_counts=True)
            mode_value = unique_vals[np.argmax(counts)]
            aggregated[f'{param}_mode'] = mode_value
    
    return aggregated


def train_M1(train_df, column_index, i, train_fraction = 1, extract_params=False):

    train_series_list = train_df.iloc[:,column_index]
    ts_df = pd.DataFrame({})
    ts_df = pd.DataFrame({
        'value': train_series_list.to_list()})
    
    feature_creation(ts_df, column_index - 1)

    if train_fraction != 1:
        ts_df = ts_df.iloc[-int(train_fraction*len(train_df)):] 

    train_x = ts_df.drop(columns=['value'], axis=1)
    train_y = ts_df['value']

    train_set = lgb.Dataset(train_x, label=train_y)

    model_dict = {}
    params_list = []
    
    for times in range(i):
        LR_list, sub_feature_list, num_leaves_list, min_data_list, max_depth_list, boost_type_list = para_num(1, 1, 1, 1, 1)

        best_params_dic = my_prequential_CV(8, 
                                        training_set=ts_df,
                                        LR_list=LR_list,
                                        sub_feature_list=sub_feature_list,
                                        num_leaves_list=num_leaves_list,
                                        min_data_list=min_data_list,
                                        max_depth_list=max_depth_list,
                                        boost_type_list=boost_type_list)

        params = {'learning_rate': best_params_dic['learning_rate'],
                    'boosting_type': best_params_dic['boosting_type'],
                    'sub_feature': best_params_dic['sub_feature'],
                    'num_leaves': best_params_dic['num_leaves'],
                    'min_data': best_params_dic['min_data'],
                    'max_depth': best_params_dic['max_depth'],
                    'verbosity': -1,
                    'feature_pre_filter': False}

        best_model = lgb.train(params, train_set)
        model_dict[f'model_number{times+1}'] = best_model

        num_estimators = best_model.current_iteration()
        if extract_params:
            best_params_dic['num_boost_round'] = num_estimators
            params_list.append(best_params_dic.copy())

    if extract_params:
        return model_dict, params_list
    else:
        return model_dict

def predict_M1_model(model_dict, test_df, column_index):

    predictions_df_1_2_test = pd.DataFrame()

    test_series_list = test_df.iloc[:, column_index]
    tst_df = pd.DataFrame({})
    tst_df = pd.DataFrame({
        'value': test_series_list.to_list()})
    
    feature_creation(tst_df, column_index - 1)

    test_x = tst_df.drop(columns=['value'], axis=1)
    test_y = tst_df['value']

    for key, model in model_dict.items():
        test_yhat = model.predict(test_x)
        predictions_df_1_2_test[key] = test_yhat[10:]

    predictions = predictions_df_1_2_test.mean(axis=1)
    final_df = test_df.iloc[10:].reset_index(drop = True)
    final_df['m_all'] = predictions

    return final_df, predictions_df_1_2_test

#########################################################################################################################


def RSS_calculator(y, y_hat):
    return np.sum((y - y_hat) ** 2)

def MAE_calculator(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def feature_creation_aecw(lag_count, df):
    for i in range(1, lag_count + 1):
        df[f'lag_{i}'] = df['label_value'].shift(i).fillna(0)
    return df

def load_m_recent_params(aggregated_params_dict, train_exposure):
    if train_exposure in aggregated_params_dict:
        params = aggregated_params_dict[train_exposure]
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


from batch_error_detection_module import BEDM
def Adaptive_ECW(label_df, lag_count, forecast_h, error_function, aggregated_params_dict, train_exposure=None, monitor_param=2):

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
    retraining_count = 0
    
    m_recent_params = load_m_recent_params(aggregated_params_dict, train_exposure) 

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
                    abs(weighted_all.iloc[i, -2] - weighted_all.iloc[i, 2]), monitor_param
                ))
            if 'GD detected' in status:
                index = status.index('GD detected')
                window_length = index - 1

            if window_length > 0:
                window = w = window_length
                temp_df = pd.DataFrame({'label_value': weighted_all.iloc[H - forecast_h:H, 2]})
                temp_df = feature_creation_aecw(lag_count, temp_df)
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
                retraining_count += 1

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

    mae = MAE_calculator(weighted_all['weighted_forecast'].values, weighted_all['label'].values)
    
    concept_1_mask = weighted_all['concept'] == 1
    concept_2_mask = weighted_all['concept'] == 2
    
    mae_concept_1 = MAE_calculator(
        weighted_all.loc[concept_1_mask, 'weighted_forecast'].values,
        weighted_all.loc[concept_1_mask, 'label'].values
    ) if concept_1_mask.any() else 0
    
    mae_concept_2 = MAE_calculator(
        weighted_all.loc[concept_2_mask, 'weighted_forecast'].values,
        weighted_all.loc[concept_2_mask, 'label'].values
    ) if concept_2_mask.any() else 0
    
    return weighted_all, retraining_count, mae, mae_concept_1, mae_concept_2

#########################################################################################################################


def run_adaptive_ecw(path, e, model_history, all_train_eval_dict, aggregated_params_dict, monitor_params):
    
    results = {}
    
    for monitor_param in monitor_params:
        df, retraining_count, mae, mae_concept_1, mae_concept_2 = Adaptive_ECW(
            label_df = all_train_eval_dict[model_history[0]]
                                    [f"train_exposure_{e}"]
                                    [f"train_evaluation"]
                                    [path],
            lag_count=10,
            forecast_h=300,
            error_function=RSS_calculator,
            aggregated_params_dict=aggregated_params_dict,
            train_exposure=e,
            monitor_param=monitor_param
        )
        
        results[monitor_param] = {
            'mae': mae,
            'mae_concept_1': mae_concept_1,
            'mae_concept_2': mae_concept_2,
            'retraining_count': retraining_count
        }
        
        del df
    
    return results

#########################################################################################################################


if __name__ == "__main__":
    model_history = {
        "m_all":1,
        "m_recent":0.3
        }

    aggregated_params_dict = {}
    all_train_eval_dict = {}

    for model_dir, history_fraction in model_history.items():
        for i in train_dict:

            if model_dir == "m_recent":
                models, params_list = train_M1(train_dict[i], 1, 10, history_fraction, extract_params=True)
                aggregated_params = aggregate_hyperparameters(params_list)
                exposure_key = int(i.split('_')[2])
                aggregated_params_dict[exposure_key] = aggregated_params
                print(f"Aggregated parameters for {i}")
            else:
                models = train_M1(train_dict[i], 1, 10, history_fraction)

            a, b = predict_M1_model(models, train_dict[i], 1)
            
            if model_dir not in all_train_eval_dict:
                all_train_eval_dict[model_dir] = {}
            
            exposure_key = i.replace('train_exposure_', '').replace('_percent', '')

            if f"train_exposure_{exposure_key}" not in all_train_eval_dict[model_dir]:
                all_train_eval_dict[model_dir][f"train_exposure_{exposure_key}"] = {"train_evaluation": {}}
            
            all_train_eval_dict[model_dir][f"train_exposure_{exposure_key}"]["train_evaluation"][f"path_results"] = a

#########################################################################################################################


    train_exposures = [0,5,10,20,30]
    model_history_list = ["m_all"]
    monitor_params = [2,3,4,5,6,7, 8, 9, 10, 11, 12,13,14,15,16,17,18,19,20]

    output_dir = BASE_DIR / f"metrics for w_l"
    output_dir.mkdir(parents=True, exist_ok=True)

    for e in train_exposures:
        print(f"\nProcessing train_exposure={e}")

        test_paths = list(all_train_eval_dict["m_all"][f"train_exposure_{e}"]["train_evaluation"].keys())

        all_results = Parallel(
            n_jobs=N_JOBS,
            backend="loky"
        )(
            delayed(run_adaptive_ecw)(
                path, e, model_history_list, all_train_eval_dict, aggregated_params_dict, monitor_params
            )
            for path in test_paths
        )

        all_metrics = {param: {'mae': [], 'mae_concept_1': [], 'mae_concept_2': [], 'retraining_count': []} for param in monitor_params}
        
        for path_results in all_results:
            for param in monitor_params:
                if param in path_results:
                    all_metrics[param]['mae'].append(path_results[param]['mae'])
                    all_metrics[param]['mae_concept_1'].append(path_results[param]['mae_concept_1'])
                    all_metrics[param]['mae_concept_2'].append(path_results[param]['mae_concept_2'])
                    all_metrics[param]['retraining_count'].append(path_results[param]['retraining_count'])
        
        summary = {}
        for param in monitor_params:
            summary[param] = {
                'mean_mae': np.mean(all_metrics[param]['mae']),
                'mean_mae_concept_1': np.mean(all_metrics[param]['mae_concept_1']),
                'mean_mae_concept_2': np.mean(all_metrics[param]['mae_concept_2']),
                'mean_retraining_count': np.mean(all_metrics[param]['retraining_count'])
            }
        
        summary_file = output_dir / f"exposure_{e}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nFor train exposure {e} w_l summary stored")
        
