import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

N_JOBS = 8
BASE_DIR = Path(__file__).resolve().parent

train_exposures = [0, 5, 10, 20, 30]
test_exposures  = [40, 50, 60]
model_history   = ["m_all", "m_recent"]

#########################################################################################################################

def RSS_calculator(y, y_hat):
    RSS = np.sum((y - y_hat) ** 2)
    return RSS


def ECW(error_function, window_size, drift_dfp, df1p, df2p):
    
    weighted_df = pd.DataFrame({
        'df1_predic': df1p.reset_index(drop=True),
        'df2_predic': df2p.reset_index(drop=True),
        'label': drift_dfp.iloc[-len(df1p):,1].reset_index(drop=True),
        'Concept': drift_dfp.iloc[-len(df1p):,0].reset_index(drop=True)
    })
    for i in window_size:
            weighted_df[f'size{i}'] = np.nan

    weighted_df = weighted_df.iloc[:]

    for w in range(len(window_size)):
            for r in range(len(weighted_df)):
                if r < window_size[w]:
                    weighted_df.iloc[r,4+w] = (weighted_df.iloc[r,0] + 
                                                weighted_df.iloc[r,1]) / 2
            
                else:
                    df1_error_res = error_function(
                        weighted_df.iloc[r - window_size[w]:r,2],
                        weighted_df.iloc[r - window_size[w]:r,0]
                    )
                    df2_error_res = error_function(
                        weighted_df.iloc[r - window_size[w]:r,2],
                        weighted_df.iloc[r - window_size[w]:r,1]
                    )

                    error_sum = df1_error_res + df2_error_res

                    if error_sum == 0:
                        weight1 = 0.5
                    else:
                        weight1 = df2_error_res/error_sum

                    weighted_df.iloc[r,4+w] = (weight1*weighted_df.iloc[r,0] + 
                                        (1-weight1)*weighted_df.iloc[r,1])
                    
            
    return weighted_df



#########################################################################################################################

def run_ecw_for_exposure_pair(te, e):

    print(f"ECW: test={te}, train={e}")

    m1_dir = (BASE_DIR/"m_all"/f"train_exposure_{e}_percent"/f"test_exposure_{te}_percent")
    m2_dir = (BASE_DIR/"m_recent"/f"train_exposure_{e}_percent"/f"test_exposure_{te}_percent")

    output_dir = BASE_DIR/"ecw"/f"{te}"/f"{e}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in m1_dir.glob("*.csv"):
        path_name = file.stem

        df_m1 = pd.read_csv(file)
        df_m2 = pd.read_csv(m2_dir / f"{path_name}.csv")

        df = ECW(
            error_function=RSS_calculator,
            window_size=[1],
            drift_dfp=df_m1.iloc[:, :2],
            df1p=df_m1.iloc[:, 2],
            df2p=df_m2.iloc[:, 2]
        )
        df.to_csv(output_dir / f"{path_name}.csv", index=False)

        del df
        del df_m1
        del df_m2

#########################################################################################################################
if __name__ == "__main__":

    tasks = [
        (te, e)
        for te in test_exposures
        for e in train_exposures
    ]

    Parallel(
        n_jobs=N_JOBS,
        backend="loky"
    )(
        delayed(run_ecw_for_exposure_pair)(te, e)
        for te, e in tasks
    )


