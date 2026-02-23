from numpy import array
import numpy as np
import random
import copy


class BEDM():
    def __init__(self, warning_level_param, concept_drift_param):
        """
        Batch Error Detection Module (BEDM) inspired by Reactive DDM for Gradual Concept Drift Problems
        

        Parameters :
        
        warning_level_param : parameter for triggering warning level signal
        concept_drift_param : parameter for triggering concept drift detection signal
        """
        self.w = warning_level_param
        self.c = concept_drift_param
        self.min_mean = 0
        self.min_std = 0
        self.nothing = 'Nothing'
        self.warning = 'Warning'
        self.warning_gd = 'GD detected'
        self.change = 'Change'
        self.warning_level = 0

    def store_concept(self, min_mean, min_std, error):
        self.min_mean = min_mean
        self.min_std = min_std
        self.error = error
        # print(f'stored values: mean = {self.min_mean}, std = {self.min_std}')

    def monitor(self, error, warning_level):
        """
        Custom Drift Detection Method (DDM) inspired by Reactive DDM for Gradual Concept Drift Problems
        Parameters :
        
        error : collection of individual observation errors to monitor model accuracy
        warning_level : number of consecutive warning_level signals to trigger gradual drift detection
        """
        new_dist = copy.deepcopy(self.error)
        new_dist.append(error)
        std = np.std(new_dist)
        
        if (error + std >= self.min_mean + (self.c * self.min_std)):
            self.warning_level = 0
            return self.change
        elif (error + std >= self.min_mean + (self.w * self.min_std)):
            self.warning_level += 1
            if (self.warning_level >= warning_level) :
                return self.warning_gd
            else:
                return self.warning
        else:
            self.warning_level = 0
            return self.nothing
        
def main():
    random.seed(1)
    dist1 = array([random.uniform(0,2) for i in range(100)])

    bedm = BEDM(2, 3)
    bedm.store_concept(np.mean(dist1), np.std(dist1), list(dist1))

    for e in range(100):
        if (e < 20):
            error = random.uniform(0,2)
        
        elif (e < 40):
            error = random.uniform(1,2)
        
        elif (e < 60):
            error = random.uniform(2,3)

        elif (e < 80):
            error = random.uniform(0,2)

        else:
            error = random.uniform(2,4)

        print("[",e,"]", error)
        print(bedm.monitor(error, limit=3))

if __name__ == "__main__":
    main()