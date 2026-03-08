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
ECW (Ensemble Combination Weighting) – offline ensemble method
A-ECW (Adaptive ECW) – extended framework with periodic retraining



# Statistical Evaluation

Model performance was evaluated using: 
forecasting error metrics
non-parametric hypothesis testing
repeated simulation experiments



# Insights

Forecast accuracy does not improve monotonically with training exposure.
Excessive retraining can reduce model stability. A trade-off exists between adaptation speed and model robustness.

This highlights the importance of balancing adaptation and stability in evolving environments.
