import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

X = cr_loan.drop('loan_status', axis = 1)
y = cr_loan[['loan_status']]

clf_logistic = LogisticRegression(solve = 'lbfgs')
clf_logistic.fit(training_colums, np.ravel(training_labels))



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 123)

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate']]
y = cr_loan_clean[['loan_status']]

# Create and fit a logistic regression model
clf_logistic_single = LogisticRegression()
clf_logistic_single.fit(X, np.ravel(y))

# Print the parameters of the model
print(clf_logistic_single.get_params())

# Print the intercept of the model
print(clf_logistic_single.intercept_)

# Create X data for the model
X_multi = cr_loan_clean[['loan_int_rate','person_emp_length']]

# Create a set of y data for training
y = cr_loan_clean[['loan_status']]

# Create and train a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X, np.ravel(y))

# Print the intercept of the model
print(clf_logistic_multi.intercept_)

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate','person_emp_length','person_income']]
y = cr_loan_clean[['loan_status']]

# Use test_train_split to create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.4, random_state=123)

# Create and fit the logistic regression model
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Print the models coefficients
print(clf_logistic.coef_)

# Model Intercept

array([-3.30582292e-10])

# Coefficients for ['loan_int_rate', 'person_emp_length', ' person_income']
array([[1.28517496e-09, -2.27622202e-09, -2.17211991e-05]])

# Calculating probabilty of default
int_coef_sum = -3.3e-10 + (1.29e-09*loan_int_rate) + (-2.28e-09*person_emp_length) + (-2.17e-05*person_income)
prob_default = 1/(1 + np.exp(-int_coef_sum))
prob_nondefault = 1 - (1/(1 + np.exp(-int_coef_sum)))

# Intercept 
intercept = -1.02
# Coefficient for employment length
person_emp_length_coef = -0.056

cr_loan_clean['loan_intent']

#separate the numeric columns
ced_num = cr_loan.select_dtypes(exclude = ['object'])
# separate non-numeric columns
cred_cat = cr_loan.select_dtypes(include = ['object'])
# One-hot necode the non-numeric columns only
cred_cat_onehot = pd.get_dummies(cred_cat)
# Union the numeric columns with the one-hot encoded columns
cr_loan = pd.concat([cred_num,cred_cat_onehot],axis = 1)

# Train the model
clf_logistic.fit(X_train, np.ravel(y_train))
# Predict using the model
clf_logistic.predict_proba(X_test)

# Print the first five rows of each training set
print(X1_train.head())
print(X2_train.head())

# Create and train a model on the first training data
clf_logistic1 = LogisticRegression(solver='lbfgs').fit(X1_train, np.ravel(y_train))

# Create and train a model on the second training data
clf_logistic2 = LogisticRegression(solver='lbfgs').fit(X2_train, np.ravel(y_train))

# Print the coefficients of each model
print(clf_logistic1.coef_)
print(clf_logistic2.coef_)

# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# Print the columns in the new data set
print(cr_loan_prep.columns)

# Check the accuarcy against the test data
clf_logistic1.score(X_test, y_test)

fallout, sensistivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout,sensitivity, color = 'darkorange')

fallout, sensistivity, thesholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')

pred = clf_logistic.predict_proba(X_test)
pred_df = pd.DataFrame(preds[:,1], columns = ['prob_dault'])
preds_df['loan_satus'] = preds_df['prob_default'].apply(lambda x: 1 if > 0.5 else 0)

from sklearn.metrics import classification_report
classification_report(y_test, preds_df['loan_status'], target_names = target_names)

fromsklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, preds_df['loan_status'])[1][1]

# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# Reassign loan status based on the threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the row counts for each loan status
print(preds_df['loan_status'].value_counts())

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))

# Print all the non-average values from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status']))

# Print all the non-average values from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1])

# Create predictions and store them in a variable
preds = clf_logistic.predict_proba(X_test)

# Print the accuracy score the model
print(clf_logistic.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_default)

# Set the threshold for defaults to 0.5
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))

# Set the threshold for defaults to 0.4
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

print(confusion_matrix(y_test,preds_df['loan_status']))

tp/tp + fn