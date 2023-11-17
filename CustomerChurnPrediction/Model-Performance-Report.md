# Customer Churn Prediction: Predicting Churn/Retention for Banking Customers via Machine Learning

## Overview of the Analysis

* The goal of this analysis is to build a machine learning model that effectively predicts which customers of the bank are likely to churn/'exit' as opposed to be retained as customers. Such a model would help to minimize losses and maximize profits for the bank through targeted churn-mitigation efforts.
* The input/feature data used for this analysis contains key factors/features about each customer, including some unuseful identifiation data as well as several other data points which seem likely to help with predicting outcomes: credit score, geography/country, gender, age, tenure,	balance, number of products, whether or not they have a credit card, their estimated salary, etc.
* The aim is to predict whether each customer will churn/exit the bank vs staying a customer / being retained. That binary outcome, i.e. "Exited" = 0 or 1, is the label/outcome that we are trying to accurately predict.
* The input dataset on bank customers and resulting outcomes (features + labels together) was loaded into Python dataframes and processed, label vs outcome data were separated, and training and testing datasets were randomly sampled for model training+evaluation.

## Model Performance Results/Options

* Model Performance Comparison matrix provided via spreadsheet here: https://docs.google.com/spreadsheets/d/1Oz8iq_hdhnD2_rl5po1pvDU0JKSyyCPYsfBUAPy-4Sg/edit#gid=0

## Summary/Recommendations

* Initial logistic regression models with manual optimization showed ok performance, with a maximum accuracy of just over 71%.
* Manually-optimized neural network models fit using TensorFlow and manual selection of model parameters were able to improve predictive accuracy significantly, with a few of the options showing close to 80% accuracy.
* Auto-optimization via Keras Tuner was able to improve performance even further, providing 2 models/options with 86+% accuracy.
* One of those models is relatively simple with only a single layer so that is the recommended model for the bank to put into use, even though it has very slightly lower accuracy than one of the other options, because it should be more efficient from a resource/computing perspective.
* Given additional time/resources it would be ideal to try additional changes to the input dataset to see if accuracy can be improved further, including digging deeper into the distributions of values for each feature/input variable in order to identify and remove potential outlier rows/datapoints. It would also make sense to bring additional data into the analysis, if available, to see if that can help improve the predictive power of the resulting models. It would also make sense to try additional deep learning model approaches, even though the bank will likely be happy with the initial predictive power of the current recommended model.