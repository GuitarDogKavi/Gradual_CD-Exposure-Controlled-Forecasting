# Evaluating the Impact of Training Exposure on Forecasting under Gradual Concept Drift

exposure - proportion of observations from the new data generating function


This project investigates forecasting performance under gradual concept drift in time series environments. 
The study explores the impact of different training exposures on forecasting by adopting an existing offline framework and a novel online adaptive extension under gradual concept drift.
A simulation framework was developed to control training exposure  and analyze model performance under evolving data distributions.



# Research Motivation

In many real-world forecasting systems, the data generating process evolves over time. This phenomenon is known as Concept Drift.
Traditional forecasting models assume a stationary environment, which can lead to performance degradation when the underlying process shifts at different drift progression rates.
This project aims to evaluate how different retraining strategies (trainng exposure) affect forecasting accuracy under gradual drift.



# Key Research Questions

How does training exposure affect forecasting accuracy under gradual drift?
Can adaptive retraining outperform offline ensemble methods?



# Methodology

Simulation Framework - A controlled simulation environment was designed to generate time series data with gradual concept drift.



# Models Evaluated

Two main approaches were compared:
ECW (Error Contribution Weighting) – offline ensemble method (Liu et al., 2023)
A-ECW (Adaptive ECW) – extended framework with periodic retraining 
BEDM (Batchwise Error Detection Module) - triggers batchwise model retraining in A-ECW by monitoring error statistics(inspired by DDM (Gama et al., 2004), RDDM (Barros et al., 2017) and DDM for Regression(Cavalcante & Oliveira, 2015))



# Statistical Evaluation

Model performance was evaluated using: 
forecasting error metrics
non-parametric hypothesis testing
repeated simulation experiments



# Insights

Forecast accuracy does not improve monotonically with training exposure.
Excessive retraining can reduce model stability. A trade-off exists between adaptation speed and model robustness.

This highlights the importance of balancing adaptation and stability in evolving environments.



# References
Liu, Z., Godahewa, R., Bandara, K., & Bergmeir, C. (2023). Handling Concept Drift in Global Time Series Forecasting (arXiv:2304.01512). arXiv. https://doi.org/10.48550/arXiv.2304.01512

Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). Learning with Drift Detection. In A. L. C. Bazzan & S. Labidi (Eds.), Advances in Artificial Intelligence – SBIA 2004 (Vol. 3171, pp. 286–295). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-540-28645-5_29

Barros, R. S. M., Cabral, D. R. L., Gonçalves, P. M., & Santos, S. G. T. C. (2017). RDDM: Reactive drift detection method. Expert Systems with Applications, 90, 344–355. https://doi.org/10.1016/j.eswa.2017.08.023

Cavalcante, R. C., & Oliveira, A. L. I. (2015). An approach to handle concept drift in financial time series based on Extreme Learning Machines and explicit Drift Detection. 2015 International Joint Conference on Neural Networks (IJCNN), 1–8. https://doi.org/10.1109/IJCNN.2015.7280721



