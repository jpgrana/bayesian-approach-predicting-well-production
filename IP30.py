import pandas as pd
import matplotlib.pyplot as plt


def IP30_Plot(propnum, df):
    y_oil = df[df['PROPNUM'] == propnum]['OIL'].reset_index(drop=True)
    y_gas = df[df['PROPNUM'] == propnum]['GAS'].reset_index(drop=True)
    y_water = df[df['PROPNUM'] == propnum]['WATER'].reset_index(drop=True)
    x = range(len(y_oil))
    y_oil_7 = y_oil.rolling(window=7, center=False).mean()
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y_oil, label='Oil', color='green')
    ax.plot(x, y_gas, label='Gas', color='red')
    ax.plot(x, y_water, label='Water', color='blue')
    plt.legend()
    ax.plot(x, y_oil_7, color='green', linestyle='--')
    ax.scatter(y_oil_7.idxmax(), y_oil_7.max(), marker='x', s=200, color='black')
    ax.annotate('Peak', (y_oil_7.idxmax(), y_oil_7.max()), xytext=(y_oil_7.idxmax() + 5, y_oil_7.max() + 500), arrowprops=dict(arrowstyle="->"), size=18)
    ax.set_title(propnum)
    ax.set_xlabel('Days On')
    ax.set_ylabel('Production')
    ax.set_xlim([0, len(y_oil)])
    ax.set_ylim([0, 10000])
    plt.show()

def get_days_production(col, df1, df2):
    days = []
    for label in df1[col]:
        days.append(df2[df2[col] == label].shape[0])
    df1['Days_Production'] = pd.Series(days)
    return df1['Days_Production']

def get_30day_peak(col, df1, df2):
    pass

if __name__ == '__main__':

    df = pd.read_csv('../other/frac_merge_drop.csv')

    df_daily = pd.read_excel('../other/AC_DAILY.xlsx')
    cols = df_daily.columns.tolist()
    cols = [col.replace(' ', '_') for col in cols]
    df_daily.columns = cols

    df['Days_Production'] = get_days_production('PROPNUM', df, df_daily)

    IP30_Plot('R7FF76VL8I', df_daily)
