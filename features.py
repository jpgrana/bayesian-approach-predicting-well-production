import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


df = pd.read_csv('../other/frac_merge_peak.csv')

X = df[[u'Completed_Feet', u'#_of_Stages', u'Stage_Length', u'Clusters/Stage', u'Cluster_Spacing', u'Perfs/Cluster', u'Fluid_Bbls', u'Fluid_Gal/Ft', u'Fluid_Gal/Cluster', u'Fluid_Gal/Perf', u'Prop_Lbs', u'Prop_Lbs/Ft', u'Prop_Lbs/Cluster', u'Prop_Lbs/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Avg_Rate', u'Max_Rate', u'Rate/Ft', u'Rate/Cluster', u'Rate/Perf', u'Avg_Pressure', u'Max_Pressure', u'ISIP/Ft', u'5"_SIP/Ft', u'XEC_FIELD', u'Reservoir']]
y = df[[u'OIL_Peak']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train_primary = X_train[[u'Completed_Feet', u'#_of_Stages', u'Stage_Length', u'Clusters/Stage', u'Perfs/Cluster', u'Fluid_Bbls', u'Prop_Lbs', u'XEC_FIELD', u'Reservoir']]
X_train_secondary = X_train[[u'Cluster_Spacing', u'Fluid_Gal/Ft', u'Fluid_Gal/Cluster', u'Fluid_Gal/Perf', u'Prop_Lbs/Ft', u'Prop_Lbs/Cluster', u'Prop_Lbs/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Avg_Rate', u'Max_Rate', u'Rate/Ft', u'Rate/Cluster', u'Rate/Perf', u'Avg_Pressure', u'Max_Pressure', u'ISIP/Ft', u'5"_SIP/Ft', u'XEC_FIELD', u'Reservoir']]
X_test_primary = X_test[[u'Completed_Feet', u'#_of_Stages', u'Stage_Length', u'Clusters/Stage', u'Perfs/Cluster', u'Fluid_Bbls', u'Prop_Lbs', u'XEC_FIELD', u'Reservoir']]
X_test_secondary = X_test[[u'Cluster_Spacing', u'Fluid_Gal/Ft', u'Fluid_Gal/Cluster', u'Fluid_Gal/Perf', u'Prop_Lbs/Ft', u'Prop_Lbs/Cluster', u'Prop_Lbs/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Avg_Rate', u'Max_Rate', u'Rate/Ft', u'Rate/Cluster', u'Rate/Perf', u'Avg_Pressure', u'Max_Pressure', u'ISIP/Ft', u'5"_SIP/Ft', u'XEC_FIELD', u'Reservoir']]

# Feature Extraction with Recursive Feature Elimination

# All Features
model = LinearRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X_train.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train)
rank_all = []
print 'Feature Ranking All:'
for col, rank in sorted(zip(X_train.drop(['XEC_FIELD', 'Reservoir'], axis=1).columns, fit.ranking_), key=lambda x : x[1]):
    print col, rank
    rank_all.append(col)
print '*' * 50
model.fit(X_train.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train)
print 'Train All R2: {0}'.format(model.score(X_train.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train))
print 'Test All R2: {0}'.format(model.score(X_test.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_test))
print '*' * 50
'''
Feature Ranking All:
ISIP/Ft 1
5"_SIP/Ft 2
Rate/Ft 3
Rate/Perf 4
Avg_Prop_Conc 5
Perfs/Cluster 6
Rate/Cluster 7
Clusters/Stage 8
Avg_Rate 9
Max_Prop_Conc 10
Max_Rate 11
Cluster_Spacing 12
Stage_Length 13
#_of_Stages 14
Fluid_Gal/Ft 15
Prop_Lbs/Ft 16
Prop_Lbs/Perf 17
Avg_Pressure 18
Fluid_Gal/Perf 19
Max_Pressure 20
Completed_Feet 21
Prop_Lbs/Cluster 22
Fluid_Bbls 23
Fluid_Gal/Cluster 24
Prop_Lbs 25
'''

# Primary Features
model = LinearRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X_train_primary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train)
rank_primary = []
print 'Feature Ranking Primary:'
for col, rank in sorted(zip(X_train_primary.drop(['XEC_FIELD', 'Reservoir'], axis=1).columns, fit.ranking_), key=lambda x : x[1]):
    print col, rank
    rank_primary.append(col)
print '*' * 50
model.fit(X_train_primary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train)
print 'Train Primary R2: {0}'.format(model.score(X_train_primary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train))
print 'Test Primary R2: {0}'.format(model.score(X_test_primary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_test))
print '*' * 50
'''
Feature Ranking Primary:
Clusters/Stage 1
Perfs/Cluster 2
#_of_Stages 3
Stage_Length 4
Completed_Feet 5
Fluid_Bbls 6
Prop_Lbs 7
'''

# Secondary Features
model = LinearRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X_train_secondary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train)
rank_secondary = []
print 'Feature Ranking Secondary:'
for col, rank in sorted(zip(X_train_secondary.drop(['XEC_FIELD', 'Reservoir'], axis=1).columns, fit.ranking_), key=lambda x : x[1]):
    print col, rank
    rank_secondary.append(col)
print '*' * 50
model.fit(X_train_secondary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train)
print 'Train Secondary R2: {0}'.format(model.score(X_train_secondary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_train))
print 'Test Secondary R2: {0}'.format(model.score(X_test_secondary.drop(['XEC_FIELD', 'Reservoir'], axis=1), y_test))
print '*' * 50
'''
Feature Ranking Secondary:
ISIP/Ft 1
5"_SIP/Ft 2
Rate/Ft 3
Rate/Perf 4
Avg_Prop_Conc 5
Max_Prop_Conc 6
Rate/Cluster 7
Max_Rate 8
Avg_Rate 9
Cluster_Spacing 10
Avg_Pressure 11
Prop_Lbs/Ft 12
Prop_Lbs/Perf 13
Max_Pressure 14
Fluid_Gal/Perf 15
Fluid_Gal/Ft 16
Prop_Lbs/Cluster 17
Fluid_Gal/Cluster 18
'''

# Features ordered by rank
rank = [u'Rate/Ft', u'ISIP/Ft', u'5"_SIP/Ft',  u'Avg_Prop_Conc', u'Perfs/Cluster', u'Rate/Perf', u'Rate/Cluster', u'Clusters/Stage', u'Max_Rate', u'Max_Prop_Conc', u'Avg_Rate', u'#_of_Stages', u'Cluster_Spacing', u'Stage_Length', u'Fluid_Gal/Ft', u'Completed_Feet', u'Fluid_Gal/Ft', u'Avg_Pressure', u'Prop_Lbs/Ft', u'Prop_Lbs/Perf', u'Max_Pressure', u'Prop_Lbs/Cluster', u'Fluid_Gal/Cluster', u'Fluid_Bbls', u'Prop_Lbs']

# Features selected by rank and correlation < 0.9
select = [u'Clusters/Stage', u'Perfs/Cluster', u'#_of_Stages', u'ISIP/Ft', u'Rate/Ft', u'Rate/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate/Cluster', u'Max_Rate', u'Cluster_Spacing', u'Avg_Pressure', u'Prop_Lbs/Ft', u'Prop_Lbs/Perf', u'Max_Pressure', u'Fluid_Gal/Perf', u'Fluid_Gal/Ft', u'Prop_Lbs/Cluster', u'Fluid_Gal/Cluster']

# Model score from select features
model = LinearRegression()
print '*' * 50
model.fit(X_train[select], y_train)
print 'Train Select R2: {0}'.format(model.score(X_train[select], y_train))
print 'Test Select R2: {0}'.format(model.score(X_test[select], y_test))
print '*' * 50
