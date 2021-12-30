# Credit Risk Analysis

![logo](images/module_17_logo.png)

# Overview
<img src="images/credit_risk.png" />

          
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, a bank or lending company will need to employ different techniques to train and evaluate models with unbalanced classes. 

In this project, I will be using several different methods to calculate Credit Risk and return which algorithm was the best at Predicting the Credit Risk of the Data Set I chose.  I will be using a credit card dataset from LendingClub, a peer-to-peer lending services company.  I will use tools to _Over Sample_ the data, _Under Sample_ the data, using machine learning techniques to predict whether the algorithm, with the data set chosen, produces a high percentage of true outcomes. 

For Deliverable 1, I will use Resampling Models to Predict Credit Risk.

For Deliverable 2, I will use the SMOTEENN Algorithm to Predict Credit Risk.

For Deliverable 3, I will use the Ensemble Classifiers to Predict Credit Risk.

The results of the above, will allow me to provide an analysis of which machine language algorithm worked best for Predicting the Outcome for Credit Risk. 

<img src="images/ml_algorithms.png" />

## Analysis Process
All of the processes below follow the same 'overall' steps, with different _Models_ to analyze the data.  I will be employing theScikit-learn machine learning library for Python in this project.  The steps will be as follows:
1)  Read the data into a Python DataFrame
2)  Check the balance of our target values
3)  Split the data into Training and Testing data sets (75% to Training 25% to Testing)
4)  Create the model (the algorithm you will use)
5)  Train the data set with the above Algorithm
6)  Calculate the balanced algorithm score
7)  Create a confusion matrix
8)  Create a confusion DataFrame and analyize the results

# Resources
* Data Sources: LoanStats_2019Q1.csv
* Software: Jupyter Notebook, Python 3.7, Pandas

# GitHub Application Link

<a href="https://jillibus.github.io/Credit_Risk_Analysis">Credit Risk Analysis</a>

## Deliverable 1: Using Resampling Models to Predict Credit Risk

The Data Set was a great candidate for the Resampling Algorithm, as the data was unbalanced. The values of the low_risk versus the high_risk data, were just too different to obtain an optimal prediction.
```
Counter({'low_risk': 51352, 'high_risk': 260})
Counter({'low_risk': 17118, 'high_risk': 87})
```

For my analysis, there are 2 ways to use resampling, you can _Over Sample_ and _Under Sample_. 

### Over Sampling

Using the **RandomOverSampler**, I resampled the X and Y data, and the Counter returned an equal paring.
```
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=1)

# Resample the targets
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)

Counter({'low_risk': 51352, 'high_risk': 51352})
```
This allowed me to analyize the _low_risk vs high_risk_ with even data.
The confusion matrix from Over Sampling and Classifier Report is below:
```
	          	Predicted high_risk	Predicted low_risk
Actual high_risk	54	          	33
Actual low_risk		5821	          	11297

                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.62      0.66      0.02      0.64      0.41        87
   low_risk       1.00      0.66      0.62      0.79      0.64      0.41     17118

avg / total       0.99      0.66      0.62      0.79      0.64      0.41     17205
```
Using Under Sampling, I used the **ClusterCentroids** from the Scikit-Learn package.
```
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)

Counter({'high_risk': 260, 'low_risk': 260})
```
This brought the high_risk number down to the low_risk number.
Again I used the LogisticRegression model, and the balanced_accuracy_store to predict the Credit Risk.
The Confusion Matrix and the Classifier Report for Under Sampling is below:
```
                    	Predicted high_risk		Predicted low_risk
Actual high_risk	50	          		37
Actual low_risk		9157	          		7961

                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.57      0.47      0.01      0.52      0.27        87
   low_risk       1.00      0.47      0.57      0.63      0.52      0.26     17118

avg / total       0.99      0.47      0.57      0.63      0.52      0.26     17205
```
This showed us that the _Under Sampling_ Model produced a worse outcome than the _Over Sampling_ Model.  The Sensitivty scores, were much lower in the Under Sampling Model.

## For Deliverable 2: I will use the SMOTEENN Algorithm to Predict Credit Risk.

I took the Over Sampling and Under Sampling Models and merged them to create a 3rd Model, to see if combining them would make the predictions even better for the customer.  Using the **SMOTEEN** Model from the Skikit-learn library, i followed the same steps, and resampled the data.
```
overunder = SMOTEENN(random_state=1)
X_resampled, y_resampled = overunder.fit_resample(X_train, y_train)
Counter(y_resampled)

Counter({'high_risk': 51351, 'low_risk': 46389})
```
I continued with the fit and balanced_accuracy_score and created the confusion matrix.  The results are below:
```
                    	Predicted high_risk	Predicted low_risk
Actual high_risk	62	          	25
Actual low_risk		7694	          	9424

                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.71      0.55      0.02      0.63      0.40        87
   low_risk       1.00      0.55      0.71      0.71      0.63      0.39     17118

avg / total       0.99      0.55      0.71      0.71      0.63      0.39     17205
```
This combination, is an overall improvement over the _Under Sampling_ model.  When comparing the Combination to the _Over Sampling_ Model
the **Precision** is the same, the **Sensitivity** is higher on high_risk customers, and lower on low_risk customers, the  **F1** score is again
the same on high_risk customers and a bit lower on low_risk customers.

## For Deliverable 3: I will use the Ensemble Classifiers to Predict Credit Risk.

To perform _Ensemble Classifiers_, I  again used the Skikit-Learn package.  I followed the same steps as above of
1)  Reading in the data set into a DataFrame
2)  Check the balance of our target values
3)  Split the data into Training and Testing data sets (75% to Training 25% to Testing)
4)  Create the model, and in this case I am going to use the **BalancedRandomForestClassifier** Model

```
# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
brfc_classifier = BalancedRandomForestClassifier(n_estimators=125, random_state=1)

# Fit Train Data
brfc_classifier.fit(X_train, y_train)

# Calculated the balanced accuracy score
y_pred = brfc_classifier.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```
The Prection came back as: 0.7988.  I ran the confusion matrix and printed the classification report to see if this model returned the 79.9% score.
```
			Predicted high_risk	Predicted low_risk
Actual high_risk	59			28
Actual low_risk		1379			15739

                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.04      0.68      0.92      0.08      0.79      0.61        87
   low_risk       1.00      0.92      0.68      0.96      0.79      0.64     17118

avg / total       0.99      0.92      0.68      0.95      0.79      0.64     17205
```
An addition check was to determine the top features, or other factors that were weighed into the decision to determine Credit Risk.
```
features_sorted = sorted(zip(brfc_classifier.feature_importants_, X.columns), reverse=True)

for feature in features_sorted:
    print(f'{feature[1]}: ({feature[0]})')
    
total_rec_prncp: (0.0706755269254938)
total_rec_int: (0.06371644260100186)
total_pymnt_inv: (0.061581668538716175)
total_pymnt: (0.061158162146968405)
last_pymnt_amnt: (0.05068753857790412)
int_rate: (0.02424115516126539)
out_prncp: (0.019106636722545946)
dti: (0.018555462772792814)
issue_d_Jan-2019: (0.018088314821659356)  
```

The second Ensemble Model I chose was the **Easy Ensemble AdaBoost Classifier**. The steps were the same as above, as shown below:
```
# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier

easy_Classifier = EasyEnsembleClassifier(n_estimators=125, random_state=1)

# Fit Train Data
easy_Classifier.fit(X_train, y_train)

# Calculated the balanced accuracy score
y_pred = easy_Classifier.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```
The prediction from the balanced_accuracy_score was 92.5%.  I continued with creating the confusion matrix and printing the classification report.
```
			Predicted high_risk	Predicted low_risk
Actual high_risk	79			8
Actual low_risk		980			16138

                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.07      0.91      0.94      0.14      0.93      0.85        87
   low_risk       1.00      0.94      0.91      0.97      0.93      0.86     17118

avg / total       0.99      0.94      0.91      0.97      0.93      0.86     17205

```
The comparision between the Ensemble Classifier and the Easy Ensemble AdaBoost Classifier, were quite different, in my opinion.
The overall _Precision_ total is still at 99%, with the _high_risk_ category a bit better in the AdaBoost Classifier.  The Sensitivity
in the _high_risk_ category for the AdaBoost Classifier out performed the Ensemble 91% to 68%, and the _low_risk_ category 94% to 92%.
Even the f1 scores the Easy Ensemble AdaBoost Classifier beat the Ensemble Classifier in both the _low_risk_ and _high_risk_ categories.

# Summary


















Thank you for your time and let me know if you wish to see any additional data.

Jill Hughes
