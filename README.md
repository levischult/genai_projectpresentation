# Increasing the resolution of GenCast's probabilistic weather forecasting: HRGenCast

## Overview
Machine Learning Weather Prediction (MLWP) has emerged as a promising alternative to Numerical Weather Prediction practices commonly used by forecasting entities globally. GenCast is one such model for global weather prediction at a 12 hour, 0.25 (~30km) spatiotemporal resolution. Consisting of the combination of graph-transformer architecture and a conditional diffusion model, GenCast is able to efficiently make probabilistic predictions of future weather. While these capabilities are impressive, there is room for improvement in the resolution of the model. In its current state, GenCast cannot forecast mesoscale phenomena such as thunderstorms, supercells, or squall-lines. These weather events frequently cause large amounts of damage to local communities, infrastructure, and environments. 

This project aims to improve the spatial resolution of GenCast down to kilometer scales and the temporal resolution to hourly. This cannot be done with the ERA5 dataset used to train the original GenCast model, so we would turn to the High-Resolution Rapid Refresh (HRRR) forecasting system. HRRR is a 3-km spatial resolution NWP model that incorporates radar data every 15 minutes and utilizes hourly data assimilation from the Rapid Refresh (13km) forecast model. By including these radar-enhanced weather prediction models, HRGenCast would have the capabilities to predict precipitation levels within convective systems such as thunderstorms. Since the HRRR model is restricted to the United States, HRGenCast would be limited in its geographic range. The model could potentially be applied to other geographic regions with further fine-tuning however. 

## Methodology


## Model Architecture + Data Used

## Implementation
- pseudocode

## Evaluation

## Critical Analysis
- possible problems/challenges
- next steps - generative data assimilation

## Resources/Citations
