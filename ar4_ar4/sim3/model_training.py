import pandas as pd
import itertools
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

#########################################################################################
# Train Data Loading


train_dict = {}
train_exposures = [0,5,10,20,30]

training_data_dir = BASE_DIR / "training_data"
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

print(train_dict.keys())

#########################################################################################
# Test Data Loading


all_test_dict = {}
test_exposures = [40, 50, 60]

test_data_dir = BASE_DIR / "test_data"

for e in test_exposures:
    folder = test_data_dir / f"{e}"
    temp_dict = {}
    for file in folder.glob("*.csv"):
        path_number = file.stem.split("_")[-1]
        temp_dict[f"path_{path_number}"] = pd.read_csv(file)
    
    all_test_dict[f'test_exposure_{e}_percent'] = temp_dict


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
                            for k in range(K):
                                if k == K - 1:
                                    break
                                train = training_set.iloc[:round((k + 1) * rows / K)]
                                valid = training_set.iloc[round((k + 1) * rows / K):round((k + 2) * rows / K)]
                                train_x = train.drop(columns=['value'], axis=1)
                                train_y = train['value']
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
                                ave_training_MAE = MAE

    return best_params_dic

#########################################################################################
# Feature Creation


def feature_creation(df, column_index, lag_count = 10):
    for i in range(1,lag_count+1,1):
        df[f'lag_{i}'] = df.iloc[:,column_index].shift(i).fillna(0)
    return df

#########################################################################################
# Store hyperparameters for dynamic M2 of A-ECW


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

#########################################################################################
# Model Training and Predicting


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
    tst_df = pd.DataFrame({'value': test_series_list.to_list()})
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

model_history = {
    "m_all":1,
    "m_recent":0.3
    }

#########################################################################################
# Storing Predictions and Hyperparameters to be called


for model_dir, history_fraction in model_history.items():
    output_dir = BASE_DIR / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in train_dict:
        sub_dir = output_dir / f"{i}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        if model_dir == "m_recent":
            models, params_list = train_M1(train_dict[i], 1, 10, history_fraction, extract_params=True)
            aggregated_params = aggregate_hyperparameters(params_list)
            params_file = sub_dir / "aggregated_hyperparameters.json"
            with open(params_file, 'w') as f:
                json.dump(aggregated_params, f, indent=4)
            print(f"Saved aggregated parameters for {i} to {params_file}")
        else:
            models = train_M1(train_dict[i], 1, 10, history_fraction)

        for exposure in all_test_dict:
            sub_dir1 = sub_dir / f"{exposure}"
            sub_dir1.mkdir(parents=True, exist_ok=True)

            for path in all_test_dict[exposure]:
                a,b = predict_M1_model(models, all_test_dict[exposure][path],1)
                a.to_csv(sub_dir1 / f"{path}.csv", index=False)
