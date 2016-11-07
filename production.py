import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Savitzky_Golay_Filter import savitzky_golay


def plot_production(propnum, df1, df2, window=31, poly=5):
    y_oil = df2[df2['PROPNUM'] == propnum]['OIL'].reset_index(drop=True)
    y_gas = df2[df2['PROPNUM'] == propnum]['GAS'].reset_index(drop=True)
    y_water = df2[df2['PROPNUM'] == propnum]['WATER'].reset_index(drop=True)
    x = range(len(y_oil))
    y_oil_smooth = savitzky_golay(y_oil.values, window, poly)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y_oil, label='Oil', color='green')
    ax.plot(x, y_gas, label='Gas', color='red')
    ax.plot(x, y_water, label='Water', color='blue')
    plt.legend()
    ax.plot(x, y_oil_smooth, color='green', linestyle='--')
    ax.scatter(y_oil_smooth.argmax(), y_oil_smooth.max(), marker='x', s=200, color='black')
    ax.annotate('Peak', (y_oil_smooth.argmax(), y_oil_smooth.max()), xytext=(y_oil_smooth.argmax() + 5, y_oil_smooth.max() + 500), arrowprops=dict(arrowstyle="->"), size=18)
    ax.set_title(propnum)
    ax.set_xlabel('Days On')
    ax.set_ylabel('Production')
    ax.set_xlim([0, len(y_oil)])
    ax.set_ylim([0, 8000])
    plt.show()
    # Optional save production graphs
    # path = '../other/production_plots/' + propnum + '_' + str(df1[df1['PROPNUM'] == propnum]['Days_Production'].values[0]) + 'DaysOn_' + str(y_oil_smooth.argmax()) + 'DaysPeak'
    # plt.savefig(path, dpi=400)
    # plt.close()

def get_days_production(col, df1, df2):
    days = []
    for label in df1[col]:
        days.append(df2[df2[col] == label].shape[0])
    df1['Days_Production'] = pd.Series(days)
    return df1['Days_Production']

def get_peak_production(col, df1, df2, ftype='OIL', window=31, poly=5):
    peaks = []
    days_to_peak = []
    for label in df1[col]:
        data = df2[df2[col] == label][ftype].reset_index(drop=True)
        data_smooth = savitzky_golay(data.values, window, poly)
        peaks.append(data_smooth.max())
        days_to_peak.append(data_smooth.argmax())
    df_peak = pd.Series(peaks)
    df_days_to_peak = pd.Series(days_to_peak)
    return df_peak, df_days_to_peak


if __name__ == '__main__':

    df = pd.read_csv('../other/frac_merge.csv')

    df_daily = pd.read_excel('../other/AC_DAILY.xlsx')
    cols = df_daily.columns.tolist()
    cols = [col.replace(' ', '_') for col in cols]
    df_daily.columns = cols

    df['Days_Production'] = get_days_production('PROPNUM', df, df_daily)
    df['OIL_Peak'], df['OIL_Days_to_Peak'] = get_peak_production('PROPNUM', df, df_daily)
    df['ISIP/Ft'].replace(0.0, df['ISIP/Ft'].mean(), inplace=True)
    df['5"_SIP/Ft'].replace(0.0, df['5"_SIP/Ft'].mean(), inplace=True)
    df.to_csv('../other/frac_merge_peak.csv', index=False)

    # Production graph examples
    # plot_production('R9EKGTQ2EH', df, df_daily) # 26 days production
    # plot_production('R7FF76VL8I', df, df_daily) # 53 days production
    # plot_production('L28IBM1H37', df, df_daily) # 2459 days production
    # plot_production('PAVGTJAB8R', df, df_daily) # 155 days peak
    
    # Optional save production graphs
    # for propnum in df['PROPNUM']:
    #     plot_production(propnum, df, df_daily)
