import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


df = pd.read_csv('../other/frac_merge_peak.csv')

X = df[[u'Completed_Feet', u'#_of_Stages', u'Stage_Length', u'Clusters/Stage', u'Cluster_Spacing', u'Perfs/Cluster', u'Fluid_Bbls', u'Fluid_Gal/Ft', u'Fluid_Gal/Cluster', u'Fluid_Gal/Perf', u'Prop_Lbs', u'Prop_Lbs/Ft', u'Prop_Lbs/Cluster', u'Prop_Lbs/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Avg_Rate', u'Max_Rate', u'Rate/Ft', u'Rate/Cluster', u'Rate/Perf', u'Avg_Pressure', u'Max_Pressure', u'ISIP/Ft', u'5"_SIP/Ft']]
X_primary = df[[u'Completed_Feet', u'#_of_Stages', u'Stage_Length', u'Clusters/Stage', u'Perfs/Cluster', u'Fluid_Bbls', u'Prop_Lbs']]
X_secondary = df[[u'Cluster_Spacing', u'Fluid_Gal/Ft', u'Fluid_Gal/Cluster', u'Fluid_Gal/Perf', u'Prop_Lbs/Ft', u'Prop_Lbs/Cluster', u'Prop_Lbs/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Avg_Rate', u'Max_Rate', u'Rate/Ft', u'Rate/Cluster', u'Rate/Perf', u'Avg_Pressure', u'Max_Pressure', u'ISIP/Ft', u'5"_SIP/Ft']]
y = df[[u'OIL_Peak']]

# Feature Extraction with Recursive Feature Elimination
model = LinearRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X, y)
for col, rank in sorted(zip(X.columns, fit.ranking_), key=lambda x : x[1]):
    print col, rank
'''
Rate/Ft 1
ISIP/Ft 2
5"_SIP/Ft 3
Avg_Prop_Conc 4
Perfs/Cluster 5
Rate/Perf 6
Rate/Cluster 7
Clusters/Stage 8
Max_Rate 9
Max_Prop_Conc 10
Avg_Rate 11
#_of_Stages 12
Cluster_Spacing 13
Stage_Length 14
Fluid_Gal/Ft 15
Completed_Feet 16
Fluid_Gal/Perf 17
Avg_Pressure 18
Prop_Lbs/Ft 19
Prop_Lbs/Perf 20
Max_Pressure 21
Prop_Lbs/Cluster 22
Fluid_Gal/Cluster 23
Fluid_Bbls 24
Prop_Lbs 25
'''
model = LinearRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X_primary, y)
for col, rank in sorted(zip(X_primary.columns, fit.ranking_), key=lambda x : x[1]):
    print col, rank
'''
Clusters/Stage 1
Perfs/Cluster 2
#_of_Stages 3
Stage_Length 4
Completed_Feet 5
Fluid_Bbls 6
Prop_Lbs 7
'''
model = LinearRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X_secondary, y)
for col, rank in sorted(zip(X_secondary.columns, fit.ranking_), key=lambda x : x[1]):
    print col, rank
'''
Rate/Ft 1
ISIP/Ft 2
5"_SIP/Ft 3
Avg_Prop_Conc 4
Rate/Perf 5
Max_Prop_Conc 6
Rate/Cluster 7
Max_Rate 8
Avg_Rate 9
Prop_Lbs/Ft 10
Fluid_Gal/Ft 11
Avg_Pressure 12
Cluster_Spacing 13
Prop_Lbs/Perf 14
Fluid_Gal/Perf 15
Max_Pressure 16
Prop_Lbs/Cluster 17
Fluid_Gal/Cluster 18
'''
