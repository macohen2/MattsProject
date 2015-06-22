# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:38:10 2015

@author: MatthewCohen
"""

import sqlite3
import pandas
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
import numpy
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt

conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/TeamSeason1.sqlite')
query = """SELECT t.won as wins, g.good_team, t.o_fgm as field_goals_made, t.o_fga as field_goals_attempted,
t.o_ftm as free_throws_made, t.o_fta as free_throws_attempted, t.o_oreb as offensive_rebounds,
t.o_dreb as defensive_rebounds, t.o_reb as total_rebounds, t.o_asts as assists, t.o_pf as personal_fouls,
t.o_stl as steals, t.o_to as turnovers, t.o_3pm as three_pointers_made, t.o_3pa as three_pointers_attempted,
t.d_fgm as field_goals_allowed, t.d_fga as field_goal_attempts_allowed, t.d_reb as rebounds_allowed,
t.d_asts as assists_allowed, t.d_pf as fouls_against, t.d_3pm as three_point_makes_allowed,
((o_fgm / o_fga)*100) as field_goal_percentage, ((o_ftm / o_fta)*100) as free_throw_percentage,
((o_3pm / o_3pa)*100) as three_point_percentage, o_blk as blocks, o_pts as points, d_pts as points_against
FROM TeamSeason1 t
LEFT OUTER JOIN Good_Teams2 g ON t.team = g.team and t.year = g.year
WHERE t.year > 1980 and t.year <= 2009;"""
df = pandas.read_sql(query, conn)
conn.close

# Defining Explanatory Features
#'field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made'
explanatory_features = [col for col in df.columns if col not in ['field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made']]
explanatory_df = df[explanatory_features]
explanatory_colnames = explanatory_df.columns

# Defining Response Series
response_series = df.good_team



### FINDING CORRELATION

# first, let's create a correlation matrix diagram for the first 26 features.

# Correlation Matrix Heat Map


toChart = explanatory_df.corr()
plt.pcolor(toChart)
plt.yticks(numpy.arange(0.5, len(toChart.index), 1), toChart.index)
plt.xticks(numpy.arange(0.5, len(toChart.columns), 1), toChart.columns, rotation=-90)
plt.colorbar()
plt.show()




# Eliminating features that are very highly correlated (if any exist)
def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  numpy.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) >= 0.9].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}

no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)
print no_correlation




# Scaling data such that it is normally distributed in order to input into model and improve accuracy
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)






# Recursive feature elimination with Cross Validation

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import tree

class RandomForestsWithCoef(ensemble.RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(ensemble.RandomForestClassifier, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

rfWithCoef = RandomForestsWithCoef(n_estimators= 500)
rfe_cv = RFECV(estimator=rfWithCoef, step=1, cv=10, scoring='roc_auc', verbose = 0)
rfe_cv.fit(explanatory_df, response_series)

print "Optimal number of features :{0} of {1} considered".format(rfe_cv.n_features_,len(explanatory_df.columns))


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC_AUC)")
plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
plt.show()

features_used = explanatory_df.columns[rfe_cv.get_support()]
print features_used

final_estimator_used = rfe_cv.estimator_



# Defining New Feature Set
explanatory_features = [col for col in df.columns if col not in ['field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made', 'offensive_rebounds']]
explanatory_df = df[explanatory_features]
explanatory_colnames = explanatory_df.columns




# Predicting wins using Random Forests 
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_rf.mean()


# Grid Search for best parameters
trees_range = range(10, 300, 10)
param_grid = dict(n_estimators = trees_range)
grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(trees_range, grid_mean_scores)


best_rf_est = grid.best_estimator_
print best_rf_est.n_estimators
print grid.best_score_


# Plotting Sensitivity vs Specificity
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)
rf_probabilities = pandas.DataFrame(best_rf_est.fit(xTrain, yTrain).predict_proba(xTest))
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])

plt.figure()
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


# PREDICT ON NEW DATA ######


# Making two different groups - TRAINING and HOLDOUT
# TRAINING
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/TeamSeason1.sqlite')
query = """SELECT t.won as wins, g.good_team, t.o_fgm as field_goals_made, t.o_fga as field_goals_attempted,
t.o_ftm as free_throws_made, t.o_fta as free_throws_attempted, t.o_oreb as offensive_rebounds,
t.o_dreb as defensive_rebounds, t.o_reb as total_rebounds, t.o_asts as assists, t.o_pf as personal_fouls,
t.o_stl as steals, t.o_to as turnovers, t.o_3pm as three_pointers_made, t.o_3pa as three_pointers_attempted,
t.d_fgm as field_goals_allowed, t.d_fga as field_goal_attempts_allowed, t.d_reb as rebounds_allowed,
t.d_asts as assists_allowed, t.d_pf as fouls_against, t.d_3pm as three_point_makes_allowed,
((o_fgm / o_fga)*100) as field_goal_percentage, ((o_ftm / o_fta)*100) as free_throw_percentage,
((o_3pm / o_3pa)*100) as three_point_percentage, o_blk as blocks, o_pts as points, d_pts as points_against
FROM TeamSeason1 t
LEFT OUTER JOIN Good_Teams2 g ON t.team = g.team and t.year = g.year
WHERE t.year > 1980 and t.year < 1999;"""
df = pandas.read_sql(query, conn)
conn.close

# Defining Explanatory Features
#'field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made'
explanatory_features = [col for col in df.columns if col not in ['field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made', 'offensive_rebounds']]
explanatory_df = df[explanatory_features]
explanatory_colnames = explanatory_df.columns

# Defining Response Series
response_series = df.good_team

# Scaling data such that it is normally distributed in order to input into model and improve accuracy
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

# Predicting wins using Random Forests 
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Grid Search for best parameters
trees_range = range(10, 300, 10)
param_grid = dict(n_estimators = trees_range)
grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(trees_range, grid_mean_scores)


best_rf_est = grid.best_estimator_


# HOLDOUT
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/TeamSeason1.sqlite')
query2 = """SELECT t.won as wins, g.good_team, t.o_fgm as field_goals_made, t.o_fga as field_goals_attempted,
t.o_ftm as free_throws_made, t.o_fta as free_throws_attempted, t.o_oreb as offensive_rebounds,
t.o_dreb as defensive_rebounds, t.o_reb as total_rebounds, t.o_asts as assists, t.o_pf as personal_fouls,
t.o_stl as steals, t.o_to as turnovers, t.o_3pm as three_pointers_made, t.o_3pa as three_pointers_attempted,
t.d_fgm as field_goals_allowed, t.d_fga as field_goal_attempts_allowed, t.d_reb as rebounds_allowed,
t.d_asts as assists_allowed, t.d_pf as fouls_against, t.d_3pm as three_point_makes_allowed,
((o_fgm / o_fga)*100) as field_goal_percentage, ((o_ftm / o_fta)*100) as free_throw_percentage,
((o_3pm / o_3pa)*100) as three_point_percentage, o_blk as blocks, o_pts as points, d_pts as points_against
FROM TeamSeason1 t
LEFT OUTER JOIN Good_Teams2 g ON t.team = g.team and t.year = g.year
WHERE t.year >= 1999 and t.year <= 2009;"""
df2 = pandas.read_sql(query2, conn)
conn.close

# Defining Explanatory Features
#'field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against"
explanatory_features2 = [col for col in df2.columns if col not in ['field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made', 'offensive_rebounds']]
explanatory_df2 = df2[explanatory_features2]
explanatory_colnames2 = explanatory_df2.columns

# Defining Response Series
response_series2 = df2.good_team

# Scaling data such that it is normally distributed in order to input into model and improve accuracy
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df2)
explanatory_df2 = pandas.DataFrame(scaler.transform(explanatory_df2), columns = explanatory_df2.columns)

# Creating Prediction Object
prediction = best_rf_est.predict(explanatory_df2)

# VISUALIZE THIS

import numpy as np
import matplotlib.pyplot as plt
import pylab

y = response_series2.tolist()
yhat = prediction.tolist()

def randomize(mean):
    N = 5
    cov = [[0.02, 0.02], [0, 0.02]]
    x,y = np.random.multivariate_normal(mean, cov, N).T
    plt.scatter(x, y, s=70, alpha=0.03)
#    return x,y

for i in range(1,325):
    randomize((y[i], yhat[i]))
 #   plt.scatter(x, y, s=70, alpha=0.03)


# Accuracy

diff = (response_series2 - prediction).tolist()
accuracy = (diff.count(0)/len(diff))*100
print accuracy




# List of feature importances
importances = pandas.DataFrame(grid.best_estimator_.feature_importances_, index = explanatory_df.columns, columns =['importance'])
importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances




# Recursive feature elimination


#rfWithCoef = RandomForestsWithCoef(n_estimators= 500)
rfe = RFE(estimator=rfWithCoef, n_features_to_select=3, step=1, verbose = 0)
rfe.fit(explanatory_df, response_series)

features_used = explanatory_df.columns[rfe.get_support()]
print features_used


# Run random forests on 3 best features


conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/TeamSeason1.sqlite')
query = """SELECT t.won as wins, g.good_team, t.o_fgm as field_goals_made, t.o_fga as field_goals_attempted,
t.o_ftm as free_throws_made, t.o_fta as free_throws_attempted, t.o_oreb as offensive_rebounds,
t.o_dreb as defensive_rebounds, t.o_reb as total_rebounds, t.o_asts as assists, t.o_pf as personal_fouls,
t.o_stl as steals, t.o_to as turnovers, t.o_3pm as three_pointers_made, t.o_3pa as three_pointers_attempted,
t.d_fgm as field_goals_allowed, t.d_fga as field_goal_attempts_allowed, t.d_reb as rebounds_allowed,
t.d_asts as assists_allowed, t.d_pf as fouls_against, t.d_3pm as three_point_makes_allowed,
((o_fgm / o_fga)*100) as field_goal_percentage, ((o_ftm / o_fta)*100) as free_throw_percentage,
((o_3pm / o_3pa)*100) as three_point_percentage, o_blk as blocks, o_pts as points, d_pts as points_against
FROM TeamSeason1 t
LEFT OUTER JOIN Good_Teams2 g ON t.team = g.team and t.year = g.year
WHERE t.year > 1980 and t.year <= 2009;"""
df = pandas.read_sql(query, conn)
conn.close

# Defining Explanatory Features
#'field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made'
explanatory_features = [col for col in df.columns if col in ['defensive_rebounds', 'field_goal_percentage', 'assists_allowed']]
explanatory_df = df[explanatory_features]
explanatory_colnames = explanatory_df.columns

# Defining Response Series
response_series = df.good_team

# Scaling data such that it is normally distributed in order to input into model and improve accuracy
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)


# Predicting wins using Random Forests 
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)
print roc_scores_rf.mean()


# Grid Search for best parameters
trees_range = range(10, 300, 10)
param_grid = dict(n_estimators = trees_range)
grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(trees_range, grid_mean_scores)


best_rf_est = grid.best_estimator_
print best_rf_est.n_estimators
print grid.best_score_

















## FINDING ACCURACY OF SUBSET OF FEATURES


# Making two different groups - TRAINING and HOLDOUT
# TRAINING
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/TeamSeason1.sqlite')
query = """SELECT t.won as wins, g.good_team, t.o_fgm as field_goals_made, t.o_fga as field_goals_attempted,
t.o_ftm as free_throws_made, t.o_fta as free_throws_attempted, t.o_oreb as offensive_rebounds,
t.o_dreb as defensive_rebounds, t.o_reb as total_rebounds, t.o_asts as assists, t.o_pf as personal_fouls,
t.o_stl as steals, t.o_to as turnovers, t.o_3pm as three_pointers_made, t.o_3pa as three_pointers_attempted,
t.d_fgm as field_goals_allowed, t.d_fga as field_goal_attempts_allowed, t.d_reb as rebounds_allowed,
t.d_asts as assists_allowed, t.d_pf as fouls_against, t.d_3pm as three_point_makes_allowed,
((o_fgm / o_fga)*100) as field_goal_percentage, ((o_ftm / o_fta)*100) as free_throw_percentage,
((o_3pm / o_3pa)*100) as three_point_percentage, o_blk as blocks, o_pts as points, d_pts as points_against
FROM TeamSeason1 t
LEFT OUTER JOIN Good_Teams2 g ON t.team = g.team and t.year = g.year
WHERE t.year > 1980 and t.year < 1999;"""
df = pandas.read_sql(query, conn)
conn.close

# Defining Explanatory Features
#'field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against', 'free_throws_made', 'three_pointers_made'
explanatory_features = [col for col in df.columns if col in ['defensive_rebounds', 'assists_allowed', 'field_goal_percentage']]
explanatory_df = df[explanatory_features]
explanatory_colnames = explanatory_df.columns

# Defining Response Series
response_series = df.good_team

# Scaling data such that it is normally distributed in order to input into model and improve accuracy
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)

# Predicting wins using Random Forests 
rf = ensemble.RandomForestClassifier(n_estimators= 500)
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc', n_jobs = -1)

# Grid Search for best parameters
trees_range = range(10, 300, 10)
param_grid = dict(n_estimators = trees_range)
grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc', n_jobs = -1)
grid.fit(explanatory_df, response_series)
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(trees_range, grid_mean_scores)


best_rf_est = grid.best_estimator_


# HOLDOUT
conn = sqlite3.connect('/Users/MatthewCohen/Documents/SQLite/TeamSeason1.sqlite')
query2 = """SELECT t.won as wins, g.good_team, t.o_fgm as field_goals_made, t.o_fga as field_goals_attempted,
t.o_ftm as free_throws_made, t.o_fta as free_throws_attempted, t.o_oreb as offensive_rebounds,
t.o_dreb as defensive_rebounds, t.o_reb as total_rebounds, t.o_asts as assists, t.o_pf as personal_fouls,
t.o_stl as steals, t.o_to as turnovers, t.o_3pm as three_pointers_made, t.o_3pa as three_pointers_attempted,
t.d_fgm as field_goals_allowed, t.d_fga as field_goal_attempts_allowed, t.d_reb as rebounds_allowed,
t.d_asts as assists_allowed, t.d_pf as fouls_against, t.d_3pm as three_point_makes_allowed,
((o_fgm / o_fga)*100) as field_goal_percentage, ((o_ftm / o_fta)*100) as free_throw_percentage,
((o_3pm / o_3pa)*100) as three_point_percentage, o_blk as blocks, o_pts as points, d_pts as points_against
FROM TeamSeason1 t
LEFT OUTER JOIN Good_Teams2 g ON t.team = g.team and t.year = g.year
WHERE t.year >= 1999 and t.year <= 2009;"""
df2 = pandas.read_sql(query2, conn)
conn.close

# Defining Explanatory Features
#'field_goals_made', 'field_goals_allowed', 'good_team', 'wins', 'points', 'points_against"
explanatory_features2 = [col for col in df2.columns if col in ['defensive_rebounds', 'assists_allowed', 'field_goal_percentage']]
explanatory_df2 = df2[explanatory_features2]
explanatory_colnames2 = explanatory_df2.columns

# Defining Response Series
response_series2 = df2.good_team

# Scaling data such that it is normally distributed in order to input into model and improve accuracy
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df2)
explanatory_df2 = pandas.DataFrame(scaler.transform(explanatory_df2), columns = explanatory_df2.columns)


prediction = best_rf_est.predict(explanatory_df2)

# VISUALIZE THIS

import numpy as np
import matplotlib.pyplot as plt
import pylab

y = response_series2.tolist()
yhat = prediction.tolist()

def randomize(mean):
    N = 5
    cov = [[0.02, 0.02], [0, 0.02]]
    x,y = np.random.multivariate_normal(mean, cov, N).T
    plt.scatter(x, y, s=70, alpha=0.03)
#    return x,y

for i in range(1,325):
    randomize((y[i], yhat[i]))
 #   plt.scatter(x, y, s=70, alpha=0.03)


# Accuracy

diff = (response_series2 - prediction).tolist()
accuracy = (diff.count(0)/len(diff))*100
print accuracy



