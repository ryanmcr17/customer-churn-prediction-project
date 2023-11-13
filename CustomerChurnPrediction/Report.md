# Credit Risk / Loan Health Analysis: ML Model Options and Recommendations

## Overview of the Analysis

* The goal of this analysis is to identify a credit-risk/loan-health prediction model that effectively flags high-risk loans for manual review, in order to minimize losses / maximize profits for the bank's credit/lending practice.
* The input data used for this analysis contains key factors/features about customer loans including: loan size, interest rate, borrower income, borrower debt-to-income ratio, borrower number of accounts, borrower total debt, as well as 'derogatory marks'.
* The goal is to predict the eventual status of each loan based on those factors/features, where loan status can be either 'healthy' or 'at risk'.
* The existing data on customer loans and loan-health outcomes was loaded and parsed into Python dataframes, including randomly sampling the data into separate training and testing datasets as well as separating the input/feature data from the outcome/result data.
* Models were fit using both standard logistic regression as well as logistic regression applied to over-sampled data (in order to reduce negative impact from the dataset being unbalanced toward healthy loan outcomes), and results/effectiveness of the two resulting models were compared.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* ML Model 1 (standard logistic regression applied to original data):
  * Accuracy (Balanced Accuracy Score): 95.2%
  * Precision:
    * Healthy Loans: 1.00
    * High-risk Loans: 0.85
  * Recall: 
    * Healthy Loans: 0.99
    * High-risk Loans: 0.91
  * F1-Score:
    * Healthy Loans: 1.00
    * High-risk Loans: 0.88

* ML Model 2 (standard logistic regression applied to over-sampled data, to account for the unbalanced dataset and the goal of minimizing under-flagging of high-risk loans):
  * Accuracy (Balanced Accuracy Score): 99.4%
  * Precision:
    * Healthy Loans: 1.00
    * High-risk Loans: 0.84
  * Recall: 
    * Healthy Loans: 0.99
    * High-risk Loans: 0.99
  * F1-Score:
    * Healthy Loans: 1.00
    * High-risk Loans: 0.91

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* While the original logistic regression model is quite effective at predicting both healthy and high-risk loans, refitting a logistic regression model to the re-/over-sampled data produces a solidly more reliable model/prediction with higher effectiveness scores overall, especially for predicting high-risk loans.
* From a business perspective this over-sampling-based model seems likely to be much more ideal, as the  model 'missed' 52 fewer high-risk loans (just 1/14 (or ~7%) as many high-risk loans as the original model 'missed'), while only mistakenly predicting 14 additional healthy loans as being high-risk compared to the original model. This new model seems likely to be much more favorable from a profitabilty perspective and I would likely recommend that the bank put it into use for flagging potentially high-risk loans for manual/human review (though again it does depend on the cost of manual review vs the benefit of identifying high-risk loans early).
