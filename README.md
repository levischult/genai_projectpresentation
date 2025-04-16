# Increasing the resolution of GenCast's probabilistic weather forecasting: HRGenCast

## Overview
Machine Learning Weather Prediction (MLWP) has emerged as a promising alternative to Numerical Weather Prediction practices commonly used by forecasting entities globally. GenCast is one such model for global weather prediction at a 12 hour, 0.25 (~30km) spatiotemporal resolution. Consisting of the combination of graph-transformer architecture and a conditional diffusion model, GenCast is able to efficiently make probabilistic predictions of future weather. While these capabilities are impressive, there is room for improvement in the resolution of the model. In its current state, GenCast cannot forecast mesoscale phenomena such as thunderstorms, supercells, or squall-lines. These weather events frequently cause large amounts of damage to local communities, infrastructure, and environments. 

This project aims to improve the spatial resolution of GenCast down to kilometer scales and the temporal resolution to hourly. This task is considerably more difficult than the synoptic scale weather GenCast predicted for two main reasons:

1. On larger forecasting scales, hydrostatic balance (where upward pressure matches gravitational pull) is reached and models are predicting two dimensional turbulence on the globe, which can be predicted with a lead time of two weeks. At km scales, atmospheric dynamics are not in hydrostatic equilibrium (convecting thunderstorms/supercells etc). This becomes 3D turbulence which is difficult to predict.


2. Existing high-resolution weather forecasts are at hourly temporal cadence. This limits the training HRGenCast can receive, as only 10-20km scale dynamics are predictable with hourly resolution. In short, realistic km-scale convective activities will not be captured by any model trained with this level of temporal resolution.

This cannot be done with the ERA5 dataset used to train the original GenCast model, so we would turn to the High-Resolution Rapid Refresh (HRRR) forecasting system. HRRR is a 3-km spatial resolution NWP model that incorporates radar data every 15 minutes and utilizes hourly data assimilation from the Rapid Refresh (13km) forecast model. By including these radar-enhanced weather prediction models, HRGenCast would have the capabilities to predict precipitation levels within convective systems such as thunderstorms. Since the HRRR model is restricted to the United States, HRGenCast would be limited in its geographic range. The model could potentially be applied to other geographic regions with further fine-tuning however. 

## Methodology
While HRGenCast would be limited to the continental US (CONUS), boundary conditions would be supplied via the Rapid Refresh model. We would initially begin by focusing on a small region of CONUS, such as Tennessee, to explore capabilities and difficulties in an easier testbed. Once desired performance was reached, we would expand to CONUS with the design considerations informed by the regional model. Some of these design attributes that need investigation before a full CONUS model are the following:
- The original GenCast model had approximately 25 grid points per mesh node. To achieve this same coverage over CONUS (HRRRv4: 1799 x 1059 = 1905141 points), we would need approximately 76000 mesh nodes. Compared to the 41k mesh nodes in the original GenCast this is a nearly factor of two increase. It is not guaranteed, however, that the level of resolution for 30km forecasting is adequate for kilometer scale processes.
- GenCast's sparse graph transformer defined a neighborhood for each mesh node's self attention through a 32-khop radius. The coarse resolution of GenCast 

## Model Architecture + Data Used

## Implementation
- pseudocode

## Evaluation

## Critical Analysis
- possible problems/challenges
- next steps - generative data assimilation

## Resources/Citations
- gencast
- HRRR
- HRRRv4
- Rapid Refresh
- state estimation/ SDA
- MLWP in observation space
- km scale StormCast
