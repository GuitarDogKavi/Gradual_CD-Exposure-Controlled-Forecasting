from random import choices
from random import sample
import pandas as pd
from pathlib import Path
import itertools

BASE_DIR = Path(__file__).resolve().parent

def exposure_percent(alpha_values, indicator_concept):

    """
    Calculate the percentage points from the relevant concept in the indicators
    
    Parameters
    ----------
    alpha_values : list
        simulated list of concept indicator indexes
    indicator_concept : int 1 or 2
        relevant indicator whose percentage is needed to be calculated
    
    Returns
    -------
    The fraction of the requested indicator in the list
    """
    count  = 0

    for i in alpha_values:
        if i == indicator_concept:
            count += 1

    return count/len(alpha_values)

def sim(ts_length, exposure_level):
    """
    Simulate the indicator indexes of a compound time series undergoing gradual concept drift
    
    Parameters
    ----------
    ts_length : int
        required length of the simulated time series
    exposure_level : int
        Controls the speed of the drift. Higher values cause faster drift.

    Returns
    -------
    list of length ts_length
        - with elements as 1 or 2 indicating the relevant concept
    """
    alpha_list=[]

    for i in range(ts_length):
        w2 = min((i/ts_length)*exposure_level,1)
        w1 = 1- w2

        alpha=choices([1,2], 
                      weights=[w1,w2],
                      k=1)
        alpha_list.append(alpha)
    alpha_list=list(itertools.chain.from_iterable(alpha_list))
    
    return alpha_list


def simulation(ts1, ts2, ts_length, train_ts_percent, exposure_threshold, exposure_weight, iterations):

    """
    Simulates a gradual concept-drift time series by mixing two regimes.

    Parameters
    ----------
    ts1 : array-like
        Time series values corresponding to concept 1.
    ts2 : array-like
        Time series values corresponding to concept 2.

    ts_length : int
        Total length of the simulated compound time series.
    train_ts_percent : float
        Percentage (0–100) of the series considered as the training window
        when matching the exposure threshold.

    exposure_threshold : int
        Target exposure percentage (0–100) of concept 2 in the training window.
    exposure_weight : int
        Controls the speed of the drift. Higher values cause faster drift.

    iterations: int
        Number of cycles to search for the relevant series


    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'alphas' : regime indicator (1 or 2)
        - 'series' : simulated time series values


    Examples
    --------
    1) simulation(ts_1,ts_2,5000,40,40,2,100) 
       returns a compound timeseries where occurence of concept 2 is 40% in the train set

    2) simulation(ts_1,ts_2,5000,40,20,1,100) 
    returns a compound timeseries where occurence of concept 2 is 20% in the train set

    """

    ts_index = None


    for i in range(iterations):

        indexes = sim(ts_length, exposure_weight)
        exposure_value = exposure_percent(indexes[:int((train_ts_percent*ts_length)/100)],2)*100

        if abs(exposure_value - exposure_threshold) < 0.001:
            ts_index = indexes
            break
        
    
    if ts_index == None:
        raise ValueError("Could not achieve target exposure; adjust parameters.")
    
    else: 
        j = k = 0
        ts = []

        for l in ts_index:
            if l == 1:
                ts.append(ts1.iloc[j])
                j += 1
            else:
                ts.append(ts2.iloc[k])
                k += 1


        df = pd.DataFrame({
            'alphas' : ts_index,
            'series' : ts
        })

        return df

concepts_dir = BASE_DIR / "Concepts_Realisations"
concept_files = sorted(concepts_dir.glob("*.csv"))

concept_data = {f.stem: pd.read_csv(f) for f in concept_files}

t1 = concept_data["Concept_1"]
t2 = concept_data["Concept_2"]

cols = range(t2.shape[1])
selected_col = sample(cols, 1)[0]
remaining_cols = []
for c in cols:
    if c != selected_col:
     remaining_cols.append(c)
    else:
       continue

print(f'training_realisation_{t1.columns[selected_col]}')
print(f'test_realisation_{[t1.columns[c] for c in remaining_cols]}')


#Simulate Training Data with limited exposures

train_exposures = [0,5,10,20,30]
output_dir = BASE_DIR / "training_data"
output_dir.mkdir(parents=True, exist_ok=True)

for i in train_exposures:
    cts = simulation(t1.iloc[:,selected_col], t2.iloc[:,selected_col], 2000, 100, i, (2*0.01*i), 1000)
    cts.to_csv(output_dir / f"cts_{list(t1.columns)[selected_col]}_exposure_{i}%.csv", index=False)


#Simulate Test Data with different exposures

test_exposures = [40, 50, 60] 
output_dir = BASE_DIR / "test_data"
output_dir.mkdir(parents=True, exist_ok=True)

for i in test_exposures: 

    sub_dir = output_dir / f"{i}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    for path in remaining_cols: 
        drift_rate = 2*0.01*i 
        
        if drift_rate*(100/100) <= 1: 
            cts = simulation(t1.iloc[10:,path], t2.iloc[10:,path], 6000, 100, i, drift_rate, 1000) 
        else: 
            drift_rate = 1/((1-(0.01*i))*2) 
            cts = simulation(t1.iloc[10:,path], t2.iloc[10:,path], 6000, 100, i, drift_rate, 1000) 

        initial_df = pd.DataFrame({ "alphas": [1] * 10, "series": t1.iloc[:10, path].values }) 
        cts_full = pd.concat([initial_df, cts], ignore_index=True)
        
        cts_full.to_csv(sub_dir / f"cts_{list(t1.columns)[path]}.csv", index=False)