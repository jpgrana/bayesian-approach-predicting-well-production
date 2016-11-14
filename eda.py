import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    df_frac = pd.read_excel('../other/GRANA_FRAC_DATA.xlsx')
    cols = df_frac.columns.tolist()
    cols = [col.replace(' ', '_') for col in cols]
    df_frac.columns = cols

    df_prop = pd.read_excel('../other/AC_PROPERTY.xlsx')
    cols = df_prop.columns.tolist()
    cols = [col.replace(' ', '_') for col in cols]
    df_prop.columns = cols

    df_prop.drop([u'DBSKEY', u'SEQNUM', u'RES_CLASS', u'XEC_RESCAT',
            u'EIAREG_FLD', u'RESERVOIR', u'OP_NONOP',
           u'QTR_BOOK', u'MTH_BOOK', u'PLANT', u'ENG', u'EXPL_REG', u'PROD_REG',
           u'PROD_DIST', u'PROD_ENG', u'MHR_CMPNY', u'PROD_ID1', u'PROD_ID2',
           u'PROD_ID3', u'PROD_CMT1', u'BTU', u'AREA_DIFF', u'GATHERING',
           u'OIL_DIFF', u'OIL_GATH', u'NGL_DIFF', u'HP_CG_POT', u'FRCST_UPD',
           u'VALUE_IND'], inplace=True, axis=1)

    df_merge = pd.merge(left=df_frac, right=df_prop, how='left', on='RSID')

    df_eur = pd.read_excel('../other/MD_Check_DB_Gross_Values.xlsx')
    cols = df_eur.columns.tolist()
    cols = [col.replace(' ', '_') for col in cols]
    df_eur.columns = cols

    df_eur.drop([u'EMS_YRBOOK', u'RESERVOIR', u'OPERATOR', u'COUNTY',
           u'STATE', u'OP_NONOP'], inplace=True, axis=1)

    df_merge2 = pd.merge(df_merge, df_eur, how='left', on='PROPNUM')

    df_merge2.rename(columns={'Wet_Gas':'Wet_Gas_EUR', 'Dry_Gas':'Dry_Gas_EUR', 'Oil':'Oil_EUR', 'NGL':'NGL_EUR'}, inplace=True)

    df_out = df_merge2[df_merge2['Oil_EUR'].notnull()]
    df_out.to_csv('../other/frac_merge.csv', index=False)
