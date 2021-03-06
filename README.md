# vece-m5-forecasting-accuracy
 
## Description
This repository has been created as part of **Project Work in Machine Learning** exam.

It contains a solution to the **M5 Forecasting - Accuracy** competition available on Kaggle (https://www.kaggle.com/c/m5-forecasting-accuracy).


## Objective
The objective consists in forecasting daily unit sales of given products in some stores across USA for the next 28 days.

## Proposed Solution
The proposed solution consists in per-store `LightGBM` models.

Calendar, prices and sales features are provided as input. 

Recursive approach is used, namely forecasts from past weeks are fed as input and used to make forecasts of next weeks.

## Results

The solution achieved `0.57415` as private score (which corresponds to the 46th position in the private leaderboard).

## Main references
- "The M5 Competition - Competitors' Guide", available at https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx
- S. Makridakis, E. Spiliotis and V. Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions". International
Journal of Forecasting (2022), available at https://doi.org/10.1016/j.ijforecast.2021.11.013
  



