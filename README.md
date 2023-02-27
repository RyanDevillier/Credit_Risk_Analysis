
# Credit Risk Analysis

## Overview of Project
The goal of this project was to determine which of several different machine learning models most accurately predicts a loan applicant’s credit risk.  To do this, we implemented six different machine learning models to a dataset that contained dozens of different attributes about tens of thousands of loan applicants, and then we evaluated the performance of these models using metrics found in Python’s sklearn library.  To conclude this project, we discuss the performance of our models and if they are practical for predicting credit risk.

## Results
To begin our analysis, we used Pandas to clean our data and make it capable of being split into training and testing sets.  Then, we imported the sklearn library and began resampling our data and creating the machine learning models.  In every machine learning model we employed, we used data that had been resampled from the original data file.  This was done as an attempt to ease the significance imbalances present in the original data (good loans significantly outnumber risky loans).  The following is a list of the performance results from each machine learning model that was used to predict credit risk:

1. Our first machine learning model was done on data that we resampled via Naïve Random Oversampling using the RandomOverSampler from the imblearn library.  We then trained a logistic regression model with this resampled data and yielded a balanced accuracy score of approximately 0.654.  To evaluate the performance of our model further, we generated a confusion matrix and an imbalanced classification report, as shown below.

![Oversampling_LogReg_Confusion_Matrix](https://user-images.githubusercontent.com/115128743/221449443-7cac3234-d6f5-4622-8e3a-006a4172faee.png)
![Oversampling_LogReg_Classification_Report](https://user-images.githubusercontent.com/115128743/221449453-6a8399fe-33b4-4ab9-b9e1-81986af8ddb9.png)


* From the classification report, we see that this model generated a precision of about 1% relative to predicting loan applicants that were deemed high risk and nearly 100% for loan applicants that were deemed low risk.  Also, we see that the model generated recall values of approximately 65% and 66% for those deemed high risk and for those deemed low risk, respectively.  Given that the precision value and the recall value differ so widely for the high risk applicants, we yielded an F1-score of 0.02, which is very low.  However, for the low risk applicants, we have a much better F1-score of 0.79 due to the precision and recall values being much closer together for the low risk applicants.  Overall, we see that this model does not predict high risk applicants well.  



2. Next, we resampled our original data with the Synthetic Minority Oversampling Technique (SMOTE).  Upon successfully resampling our data, we split this resampled data into training and testing sets and trained another logistic regression model.  This model yielded a balanced accuracy score of approximately 0.651.  To evaluate the performance of our model further, we generated a confusion matrix and an imbalanced classification report, as shown below.

![SMOTE_Confusion_Matrix](https://user-images.githubusercontent.com/115128743/221450003-5b618139-1b0c-496f-82ab-ac5078d82225.png)
![SMOTE_Classification_Report](https://user-images.githubusercontent.com/115128743/221450014-a9b13557-992a-421c-9079-978d692c8666.png)


* From the classification report, we see that this model generated a precision of about 1% relative to predicting loan applicants that were deemed high risk and nearly 100% for loan applicants that were deemed low risk.  Also, we see that the model generated recall values of approximately 64% and 66% for those deemed high risk and for those deemed low risk, respectively.  Given that the precision value and the recall value differ so widely for the high risk applicants, we yielded an F1-score of 0.02, which is very low.  However, for the low risk applicants, we have a much better F1-score of 0.79 due to the precision and recall values being much closer together for the low risk applicants.  Overall, we see that this model performs almost exactly the same as the previous oversampled model and does not predict high risk applicants well.  



3. Attempting to combat the class imbalances present in our original data, we again resampled our data, but this time with undersampling techniques instead of oversampling techniques.  Specifically, we resampled our original data using the Cluster Centroids algorithm, and then we trained a logistic regression model using this undersampled data.  This model produced a balanced accuracy score of approximately 0.500.  To validate our model further, we generated a confusion matrix and an imbalanced classification report, as shown below.

![Undersampling_Confusion_Matrix](https://user-images.githubusercontent.com/115128743/221450028-4f6e9eec-50b9-4749-b7a6-519e0f94e5bf.png)
![Undersampling_Classification_Report](https://user-images.githubusercontent.com/115128743/221450041-835f8de9-edf5-4ef7-8fdd-7659c2b503c1.png)


* From the classification report, we see that this model generated a precision of about 1% relative to predicting loan applicants that were deemed high risk and nearly 100% for loan applicants that were deemed low risk.  Also, we see that the model generated recall values of approximately 57% and 47% for those deemed high risk and for those deemed low risk, respectively.  Given that the precision value and the recall value differ so widely for the high risk applicants, we yielded an F1-score of 0.01, which is very low.  However, for the low risk applicants, we have a much better F1-score of 0.64 due to the precision and recall values being closer together for the low risk applicants.  Again, we see that this model does not predict high risk applicants well.  



4. Now that we have tried both oversampling and undersampling techniques, we turn our attention to an algorithm that combines both of these techniques together.  Namely, we employed the SMOTEENN algorithm to resample our data, and then we trained a logistic regression model with this newly resampled data.  This logistic regression model yielded a balanced accuracy score of approximately 0.503.  We then generated a confusion matrix and an imbalanced classification report to further evaluate this model’s performance.  

![SMOTEENN_Confusion_Matrix](https://user-images.githubusercontent.com/115128743/221450049-92252a17-3776-445b-aa0b-5de77e416385.png)
![SMOTEENN_Classification_Report](https://user-images.githubusercontent.com/115128743/221450060-afb0504a-8563-445c-b41f-44fabf917530.png)


* From the classification report, we see that this model generated a precision of about 1% relative to predicting loan applicants that were deemed high risk and nearly 100% for loan applicants that were deemed low risk.  Also, we see that the model generated recall values of approximately 74% and 58% for those deemed high risk and for those deemed low risk, respectively.  Given that the precision value and the recall value differ so widely for the high risk applicants, we yielded an F1-score of 0.02, which is very low.  However, for the low risk applicants, we have a much better F1-score of 0.73 due to the precision and recall values being much closer together for the low risk applicants.  This model does not predict high risk applicants well.  



5. It is at this point that we turned our attention towards ensemble learning, which is the process of combining multiple machine learning models to produce (hopefully) more accurate predictions.  Specifically, we used the Balanced Random Forest Classifier algorithm with 100 decision trees to resample our data and to generate credit risk predictions.  This machine learning model generated a balanced accuracy score of approximately 0.788.  To further evaluate the performance of this model, we generated a confusion matrix and an imbalanced classification report, as shown below.

![Balanced_Random_Forest_Confusion_Matrix](https://user-images.githubusercontent.com/115128743/221450085-a1121f03-1ad2-4e78-964e-5178836d7ab4.png)
![Balanced_Random_Forest_Classification_Report](https://user-images.githubusercontent.com/115128743/221450091-0e6e4610-dcb7-4c2c-a19a-4de92cdbf1db.png)


* From the classification report, we see that this model generated a precision of about 4% relative to predicting loan applicants that were deemed high risk and nearly 100% for loan applicants that were deemed low risk.  Also, we see that the model generated recall values of approximately 67% and 91% for those deemed high risk and for those deemed low risk, respectively.  Given that the precision value and the recall value differ so widely for the high risk applicants, we yielded an F1-score of 0.07, which is very low.  However, for the low risk applicants, we have a much better F1-score of 0.95 due to the precision and recall values being much closer together for the low risk applicants.  Overall, we see that this model does not predict high risk applicants accurately.  



6. Finally, we employed one more ensemble learning model, the Easy Ensemble AdaBoost algorithm.  We used this algorithm with 100 decision trees to resample our original data and to generate credit risk predictions.  This model yielded a balanced accuracy score of approximately 0.925, and we have displayed the respective confusion matrix and imbalanced classification report below.

![Easy_Ensemble_Adaboost_Confusion_Matrix](https://user-images.githubusercontent.com/115128743/221450099-4e8d2d60-cd98-4dec-a420-bfdb68b417c1.png)
![Easy_Ensemble_Adaboost_Classification_Report](https://user-images.githubusercontent.com/115128743/221450115-57002aff-d501-4471-a7c1-69d76896dba5.png)


* From the classification report, we see that this model generated a precision of about 7% relative to predicting loan applicants that were deemed high risk and nearly 100% for loan applicants that were deemed low risk.  Also, we see that the model generated recall values of approximately 91% and 94% for those deemed high risk and for those deemed low risk, respectively.  Given that the precision value and the recall value differ so widely for the high risk applicants, we yielded an F1-score of 0.14, which is very low.  However, for the low risk applicants, we have a much better F1-score of 0.97 due to the precision and recall values being much closer together for the low risk applicants.  Overall, we see that this model does not predict high risk applicants well.  

## Summary
Judging from the classification reports above, we see that none of our models performed well in terms of predicting which loan applicants would be considered high-risk.  The precision and F1-score values of every model were significantly lower than they would need to be in order for the models to be practical.    Overall, the ensemble learning models performed the best, though they still generated results significantly worse than random chance.
