import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as ml
import seaborn as sns
import pymc3 as pm
import patsy as pt
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import optimize
import regex as re


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

def plot_traces(trcs, varnames=None):
    '''
    Convenience fn: plot traces with overlaid means and values
    INPUT: pymc trace
    OUTPUT: display of model coefficient distributions
    '''

    nrows = len(trcs.varnames)
    if varnames is not None:
        nrows = len(varnames)

    ax = pm.traceplot(trcs, varnames=varnames, figsize=(12,nrows*1.4)
        ,lines={k: v['mean'] for k, v in
            pm.df_summary(trcs,varnames=varnames).iterrows()}
        ,combined=True)

    # don't label the nested traces (a bit clumsy this: consider tidying)
    dfmns = pm.df_summary(trcs, varnames=varnames)['mean'].reset_index()
    dfmns.rename(columns={'index':'featval'}, inplace=True)
    dfmns = dfmns.loc[dfmns['featval'].apply(lambda x: re.search('__[1-9]{1,}', x) is None)]
    dfmns['draw'] = dfmns['featval'].apply(lambda x: re.search('__0{1}$', x) is None)
    dfmns['pos'] = np.arange(dfmns.shape[0])
    dfmns.set_index('pos', inplace=True)

    for i, r in dfmns.iterrows():
        if r['draw']:
            ax[i,0].annotate('{:.2f}'.format(r['mean']), xy=(r['mean'],0)
                    ,xycoords='data', xytext=(5,10)
                    ,textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')

def predict_test(trc, X_test, X_train, hyper=0):
    '''
    Calculate mean prediction values for test data using mean coefficient values
    INPUT: pymc trace, df test, df train, number of hyperpriors
    OUTPUT: np array of mean prediction values for test data
    '''
    coeff = pm.df_summary(trc[-500:])
    X_test_std = standardize_2sd_test(X_test[fts_num], X_train[fts_num])
    preds = []
    for i in range(len(X_test)):
        if X_test.iloc[i,:]['Reservoir_Code'] == 0:
            pred = coeff.ix[0+hyper,0] + np.dot(X_test_std.iloc[i,:].values, coeff.ix[hyper+3:-1-hyper,0].values)
        if X_test.iloc[i,:]['Reservoir_Code'] == 1:
            pred = coeff.ix[1+hyper,0] + np.dot(X_test_std.iloc[i,:].values, coeff.ix[hyper+3:-1-hyper,0].values)
        else:
            pred = coeff.ix[2+hyper,0] + np.dot(X_test_std.iloc[i,:].values, coeff.ix[hyper+3:-1-hyper,0].values)
        preds.append(pred)
    return np.array(preds)

if __name__ == '__main__':

    #Data load and setup
    df = pd.read_csv('../../other/frac_merge_peak.csv')
    df['Reservoir_Code'] = pd.Categorical(df['Reservoir']).codes
    print df.groupby(['Reservoir', 'Reservoir_Code'])['Reservoir'].count()
    X = df[[u'Clusters/Stage', u'Perfs/Cluster', u'#_of_Stages', u'ISIP/Ft', u'Rate/Ft', u'Rate/Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate/Cluster', u'Max_Rate', u'Cluster_Spacing', u'Avg_Pressure', u'Prop_Lbs/Ft', u'Prop_Lbs/Perf', u'Max_Pressure', u'Fluid_Gal/Perf', u'Fluid_Gal/Ft', u'Prop_Lbs/Cluster', u'Fluid_Gal/Cluster', u'Reservoir_Code']]
    y = df[[u'OIL_Peak']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #Define dataframe for model
    ft_endog = u'OIL_Peak'
    fts_cat = u'Reservoir_Code'
    fts_num = [u'Clusters/Stage', u'Perfs/Cluster', u'#_of_Stages', u'ISIP/Ft', u'Rate/Ft', u'Rate/Perf',
                     u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate/Cluster', u'Max_Rate', u'Avg_Pressure', u'Max_Pressure',
                     u'Fluid_Gal/Perf']
    dfs = pd.concat((y_train, X_train[fts_cat], standardize_2sd(X_train[fts_num])),1)

    #Remove '/' from column names for design matrix
    dfs.columns = [u'OIL_Peak', u'Reservoir_Code', u'Clusters_Stage', u'Perfs_Cluster', u'Num_of_Stages', u'ISIP_Ft', u'Rate_Ft', u'Rate_Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate_Cluster', u'Max_Rate', u'Avg_Pressure', u'Max_Pressure', u'Fluid_Gal_Perf']
    fts_num_rename = [u'Clusters_Stage', u'Perfs_Cluster', u'Num_of_Stages', u'ISIP_Ft', u'Rate_Ft', u'Rate_Perf', u'Avg_Prop_Conc', u'Max_Prop_Conc', u'Rate_Cluster', u'Max_Rate', u'Avg_Pressure', u'Max_Pressure', u'Fluid_Gal_Perf']

    #Create design matrix
    fml_equation = '{} ~ '.format(ft_endog) + ' + '.join(fts_num_rename)
    (y_dmat, X_dmat) = pt.dmatrices(fml_equation, dfs , return_type='dataframe', NA_action='raise')

    #Create variable to run unpooled model or read from disk
    run_res_unpooled = True

    #Run or load unpooled model
    with pm.Model() as mdl_res_unpooled:

        # define priors, use Normal
        b0 = pm.Normal('b0', mu=0, sd=100, shape=dfs['Reservoir_Code'].nunique())
        b1 = pm.Normal('b1_Clusters/Stage', mu=0, sd=100)
        b2 = pm.Normal('b2_Perfs/Cluster', mu=0, sd=100)
        b3 = pm.Normal('b3_#_of_Stages', mu=0, sd=100)
        b4 = pm.Normal('b4_ISIP/Ft', mu=0, sd=100)
        b5 = pm.Normal('b5_Rate/Ft', mu=0, sd=100)
        b6 = pm.Normal('b6_Rate/Perf', mu=0, sd=100)
        b7 = pm.Normal('b7_Avg_Prop_Conc', mu=0, sd=100)
        b8 = pm.Normal('b8_Max_Prop_Conc', mu=0, sd=100)
        b9 = pm.Normal('b9_Rate/Cluster', mu=0, sd=100)
        b10 = pm.Normal('b10_Max_Rate', mu=0, sd=100)
        b11 = pm.Normal('b11_Avg_Pressure', mu=0, sd=100)
        b12 = pm.Normal('b12_Max_Pressure', mu=0, sd=100)
        b13 = pm.Normal('b13_Fluid_Gal/Perf', mu=0, sd=100)

        # define linear model
        y =    ( b0[dfs['Reservoir_Code']] +
                 b1 * X_dmat['Clusters_Stage'] +
                 b2 * X_dmat['Perfs_Cluster'] +
                 b3 * X_dmat['Num_of_Stages'] +
                 b4 * X_dmat['ISIP_Ft'] +
                 b5 * X_dmat['Rate_Ft'] +
                 b6 * X_dmat['Rate_Perf'] +
                 b7 * X_dmat['Avg_Prop_Conc'] +
                 b8 * X_dmat['Max_Prop_Conc'] +
                 b9 * X_dmat['Rate_Cluster'] +
                 b10 * X_dmat['Max_Rate'] +
                 b11 * X_dmat['Avg_Pressure'] +
                 b12 * X_dmat['Max_Pressure'] +
                 b13 * X_dmat['Fluid_Gal_Perf'])

        ## Likelihood (sampling distribution) of observations
        epsilon = pm.HalfCauchy('epsilon', beta=10)
        likelihood = pm.Normal('likelihood', mu=y, sd=epsilon, observed=dfs[ft_endog])

        if run_res_unpooled:
            trc_res_unpooled = pm.backends.text.load('../../other/traces_txt/trc_res_unpooled')
        else:
            step = pm.NUTS()
            start = pm.find_MAP()
            trace = pm.backends.Text('../../other/traces_txt/trc_res_unpooled')
            trc_res_unpooled = pm.sample(2000, step, start, trace)

    #Run unpooled model metrics for training data
    ppc_res_unpooled = pm.sample_ppc(trc_res_unpooled[-500:], samples=500, model=mdl_res_unpooled, size=50)
    y_pred_train_res_unpooled = ppc_res_unpooled['likelihood'].mean(axis=1).mean(axis=0)
    waic_res_unpooled = pm.stats.waic(model=mdl_res_unpooled, trace=trc_res_unpooled[-500:])
    print 'Unpooled Train RMSE: {0}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train_res_unpooled)))
    print 'Unpooled Train R2: {0}'.format(r2_score(y_train, y_pred_train_res_unpooled))
    print 'Unpooled Train WAIC: {0}'.format(waic_res_unpooled)

    #Run unpooled model metrics for test data
    y_pred_test = predict_test(trc_res_unpooled, X_test, X_train, hyper=0)
    print 'Unpooled Test RMSE: {0}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print 'Unpooled Test R2: {0}'.format(r2_score(y_test, y_pred_test))

    #Create variable to run new model or read from disk
    run_res_partpooled = True

    #Run or load partpooled model
    with pm.Model() as mdl_res_partpooled:

        # define hyperpriors for intercept
        b0_mu = pm.Normal('b0_mu', mu=0, sd=100)
        b0_sd = pm.HalfCauchy('b0_sd', beta=10)

        # define priors, use Normal
        b0 = pm.Normal('b0', mu=b0_mu, sd=b0_sd, shape=dfs['Reservoir_Code'].nunique())
        b1 = pm.Normal('b1_Clusters/Stage', mu=0, sd=100)
        b2 = pm.Normal('b2_Perfs/Cluster', mu=0, sd=100)
        b3 = pm.Normal('b3_#_of_Stages', mu=0, sd=100)
        b4 = pm.Normal('b4_ISIP/Ft', mu=0, sd=100)
        b5 = pm.Normal('b5_Rate/Ft', mu=0, sd=100)
        b6 = pm.Normal('b6_Rate/Perf', mu=0, sd=100)
        b7 = pm.Normal('b7_Avg_Prop_Conc', mu=0, sd=100)
        b8 = pm.Normal('b8_Max_Prop_Conc', mu=0, sd=100)
        b9 = pm.Normal('b9_Rate/Cluster', mu=0, sd=100)
        b10 = pm.Normal('b10_Max_Rate', mu=0, sd=100)
        b11 = pm.Normal('b11_Avg_Pressure', mu=0, sd=100)
        b12 = pm.Normal('b12_Max_Pressure', mu=0, sd=100)
        b13 = pm.Normal('b13_Fluid_Gal/Perf', mu=0, sd=100)

        # define linear model
        y =    ( b0[dfs['Reservoir_Code']] +
                 b1 * X_dmat['Clusters_Stage'] +
                 b2 * X_dmat['Perfs_Cluster'] +
                 b3 * X_dmat['Num_of_Stages'] +
                 b4 * X_dmat['ISIP_Ft'] +
                 b5 * X_dmat['Rate_Ft'] +
                 b6 * X_dmat['Rate_Perf'] +
                 b7 * X_dmat['Avg_Prop_Conc'] +
                 b8 * X_dmat['Max_Prop_Conc'] +
                 b9 * X_dmat['Rate_Cluster'] +
                 b10 * X_dmat['Max_Rate'] +
                 b11 * X_dmat['Avg_Pressure'] +
                 b12 * X_dmat['Max_Pressure'] +
                 b13 * X_dmat['Fluid_Gal_Perf'])

        ## Likelihood (sampling distribution) of observations
        epsilon = pm.HalfCauchy('epsilon', beta=10)
        likelihood = pm.Normal('likelihood', mu=y, sd=epsilon, observed=dfs[ft_endog])

        if run_res_partpooled:
            trc_res_partpooled = pm.backends.text.load('../../other/traces_txt/trc_res_partpooled')
        else:
            step = pm.NUTS()
            start = pm.find_MAP()
            trace = pm.backends.Text('../../other/traces_txt/trc_res_partpooled')
            trc_res_partpooled = pm.sample(2000, step, start, trace)

    #Run part-pooled model metrics for training data
    ppc_res_partpooled = pm.sample_ppc(trc_res_partpooled[-500:], samples=500, model=mdl_res_partpooled, size=50)
    y_pred_train_res_partpooled = ppc_res_partpooled['likelihood'].mean(axis=1).mean(axis=0)
    waic_res_partpooled = pm.stats.waic(model=mdl_res_partpooled, trace=trc_res_partpooled[-500:])
    print 'Part-Pooled Train RMSE: {0}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train_res_partpooled)))
    print 'Part-Pooled Train R2: {0}'.format(r2_score(y_train, y_pred_train_res_partpooled))
    print 'Part-Pooled Train WAIC: {0}'.format(waic_res_partpooled)

    #Run part-pooled model metrics for test data
    y_pred_test_res_partpooled = predict_test(trc_res_partpooled, X_test, X_train, hyper=1)
    print 'Part-Pooled Test RMSE: {0}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test_res_partpooled)))
    print 'Part-Pooled Test R2: {0}'.format(r2_score(y_test, y_pred_test_res_partpooled))
