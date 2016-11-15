import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Define standardization functions
def standardize_2sd(df):
    '''
    Standardize training data based on 2 standard deviations
    INPUT: dataframe
    OUTPUT: dataframe
    '''
    return (df - df.mean(0)) / (2 * df.std(0))

def standardize_2sd_test(df_test, df_train):
    '''
    Standardize test data based on 2 standard deviations
    INPUT: test dataframe, train dataframe
    OUTPUT: dataframe
    '''
    return (df_test - df_train.mean(0)) / (2 * df_train.std(0))

#Define plot functions
def plot_traces(traces, retain=1000):
    '''
    Plot traces with overlaid means and values
    INPUT: pymc trace, traces to retain
    OUTPUT: display of model coefficient distributions
    '''

    ax = pm.traceplot(traces[-retain:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.df_summary(traces[-retain:]).iterrows()})

    for i, mn in enumerate(pm.df_summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')


if __name__ == '__main__':

    #Data load and setup
    df = pd.read_csv('../other/frac_merge_peak.csv')

    X = df[[u'Clusters/Stage', u'Perfs/Cluster', u'#_of_Stages', u'ISIP/Ft', u'Rate/Ft', u'Rate/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate/Cluster', u'Max_Rate', u'Avg_Pressure', u'Max_Pressure', u'Fluid_Gal/Perf']]
    y = df[[u'OIL_Peak']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #Linear regression model summary
    X_OLS_train = sm.add_constant(standardize_2sd(X_train))
    model = sm.OLS(y_train, X_OLS_train)
    model = model.fit()
    print model.summary()

    #Linear regression model scores
    model = LinearRegression(fit_intercept=True, normalize=False)
    model.fit(X_OLS_train.drop('const', axis=1), y_train)
    print 'Train R2: {0}'.format(model.score(X_OLS_train.drop('const', axis=1), y_train))
    X_OLS_test = standardize_2sd_test(X_test, X_train)
    print 'Test R2: {0}'.format(model.score(X_OLS_test, y_test))

    #Regularized Lasso model summary
    lasso = sm.OLS(y_train, X_OLS_train).fit_regularized(alpha=1, L1_wt=1)
    print lasso.summary()

    #regularized Lasso model score
    X_lasso_train = X_train[[u'Clusters/Stage', u'Perfs/Cluster', u'#_of_Stages', u'ISIP/Ft', u'Rate/Ft', u'Rate/Perf',
                     u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate/Cluster', u'Max_Rate', u'Avg_Pressure', u'Max_Pressure',
                     u'Fluid_Gal/Perf']]
    X_lasso_train_std = standardize_2sd(X_lasso_train)
    model = LinearRegression(fit_intercept=True, normalize=False)
    model.fit(X_lasso_train_std, y_train)
    print 'Lasso Train R2: {0}'.format(model.score(X_lasso_train_std, y_train))
    X_lasso_test = X_test[[u'Clusters/Stage', u'Perfs/Cluster', u'#_of_Stages', u'ISIP/Ft', u'Rate/Ft', u'Rate/Perf',
                     u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate/Cluster', u'Max_Rate', u'Avg_Pressure', u'Max_Pressure',
                     u'Fluid_Gal/Perf']]
    X_lasso_test_std = standardize_2sd_test(X_lasso_test, X_lasso_train)
    print 'Lasso Test R2: {0}'.format(model.score(X_lasso_test_std, y_test))

    #Variable to re-run model or load from disk
    run_pooled = True

    #Run PYMC unpooled model using GLM notation
    data = dict(x=X_lasso_train_std, y=y_train)

    with pm.Model() as mdl_pooled:
        pm.glm.glm('y ~ x', data, family=pm.glm.families.Normal())
        if run_pooled:
            trc_pooled = pm.backends.text.load('../other/traces_txt/trc_pooled')
        else:
            step = pm.NUTS()
            start = pm.find_MAP()
            trace = pm.backends.Text('../other/traces_txt/trc_pooled')
            trc_pooled = pm.sample(2000, njobs=1, step=step, start=start, trace=trace)

    #Run unpooled model metrics for training data
    ppc_pooled = pm.sample_ppc(trc_pooled[-1000:], samples=500, model=mdl_pooled, size=50)
    y_pred = ppc_pooled['y'].mean(0).mean(0).T
    print 'Bayesian Train RMSE: {0}'.format(np.sqrt(mean_squared_error(y_train, y_pred)))
    print 'Bayesian Train R2: {0}'.format(r2_score(y_train, y_pred))

    #Run unpooled model metrics for test data
    parm_pooled = pm.df_summary(trc_pooled[-1000:]).values
    y_pred_test = parm_pooled[0,0] + np.dot(X_lasso_test_std.values, parm_pooled[1:-1,0])
    print 'Bayesian Test RMSE: {0}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print 'Bayesian Test R2: {0}'.format(r2_score(y_test, y_pred_test))
