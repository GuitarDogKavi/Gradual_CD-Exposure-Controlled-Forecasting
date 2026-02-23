import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
output_dir = BASE_DIR / "Concepts_Realisations"
output_dir.mkdir(parents=True, exist_ok=True)

def coefficient_generator(max_root, lags, rng):
    """
    Used to generate the coefficients for AR time series
    while ensuring stationarity

    max root : 
        maximum possible root required
        need to be greater than 1.1 
    lags : 
        number of lag terms to build the structure of the AR process

    rng :
         numpy.random.Generator object for reproducibility

    ex:- Xt = -0.25*X(t-1) -0.25*X(t-2)
    lags = 2
    
    """
    if max_root <= 1.1:
        raise ValueError("AR structure is not stationary")

    abs_root = rng.uniform(1.1, max_root, lags)
    sign_val = rng.uniform(-1, 1, lags)
    signs = np.sign(sign_val)

    roots = abs_root * signs
    coefficients = np.poly(roots)

    std_coefficients = coefficients / coefficients[-1]
    return list(std_coefficients[::-1])
    

def simulate_ar_paths(coefficients, m, n, sigma=1.0, burnin=200, seed=None):

    """
    Simulate an AR process using pre-generated coefficients

    coefficients  : [1, φ1, φ2, ..., φp]
    m             : number of paths
    n             : length of final series
    sigma         : std dev of white noise
    burnin        : discard initial samples
    seed          : to randomise or reproduce gaussian white noise sampling
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(coefficients) - 1
    phi = -np.array(coefficients[1:])

    eps = np.random.normal(0, sigma, size=(m, n + burnin))
    x = np.zeros((m, n + burnin))

    for t in range(p, n + burnin):
        x[:, t] = x[:, t-p:t][:, ::-1] @ phi + eps[:, t]

    return x[:, burnin:]



def ts_generation_sample_path_simulation(max_root_1, lags_1, max_root_2, lags_2, seed, sample_paths, ts_length):

    """
    Simulate concept 1 and concept 2 with defined number of sample paths and time series length


    max_root_1             : maximum possible root required for concept 1 (need to be greater than 1.1) 
    lags_1                 : number of lag terms to build the structure of the AR process for concept 1
    max_root_2             : maximum possible root required for concept 2 (need to be greater than 1.1)
    lags_2                 : number of lag terms to build the structure of the AR process for concept 2
    seed                   : integer seed for reproducibility
    sample_paths           : number of sample paths for each concept
    ts_length              : length of each sample path

    """


    rng = np.random.default_rng(seed)

    j = 0
    
    coeff1 = coefficient_generator(max_root_1, lags_1, rng)
    coeff2 = coefficient_generator(max_root_2, lags_2, rng)
    
    ts1 = simulate_ar_paths(coeff1,sample_paths,ts_length, seed=seed +1)
    ts2 = simulate_ar_paths(coeff2,sample_paths,ts_length, seed=seed +1)

    df_c1 = pd.DataFrame(
        ts1.T,
        columns=[f"path_{i+1}" for i in range(ts1.shape[0])]
        )
    
    df_c2 = pd.DataFrame(
        ts2.T,
        columns=[f"path_{i+1}" for i in range(ts2.shape[0])]
        )
    
    df_c1.to_csv(output_dir / "Concept_1.csv", index=False)
    df_c2.to_csv(output_dir / "Concept_2.csv", index=False)

    coefficients_log = pd.DataFrame({
        "process": ["Concept_1", "Concept_2"],
        "coefficients": [coeff1, coeff2]})

    coefficients_log.to_csv(output_dir / "Concepts.csv", index=False)
    return 


ts_generation_sample_path_simulation(5,4,5,7,48,201,4010)
