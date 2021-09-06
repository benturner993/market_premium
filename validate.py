import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

def gini_table(df, pred, act):
    
    ''' returns a dataframe from which a gini coefficient can be calculated
        also can create cumulative gains curves
        pred = predicted values (output from emblem) argument required is the name of the column in df
        act = actual values (number of claims) argument required is the name of the column in df
    
        3 useful outputs
        Perc of Obs and Perc of Claims can be used to create Cumulative Gains Curves
        Gini_Area can be used to calculate the gini coefficient. Each Gini_Area is the approximate area under Cumulative
        gains curve. Feel free to change to trapezium rule in future. '''
    
    df = df[[pred, act]].sort_values(by=pred, ascending=False)
    df = df.reset_index()
    df['Cumulative Claims'] = df[act].cumsum()
    df['Perc of Obs'] = (df.index + 1) / df.shape[0]
    df['Perc of Claims'] = df['Cumulative Claims'] / df.iloc[-1]['Cumulative Claims']
    df['gini_area'] = df['Perc of Claims'] / df.shape[0]
    return df

def calc_gini(df, pred, act):
    
    ''' uses output from gini_table to calculate a gini coefficient. 
        model = column name of modelled values you wish to calculate gini coefficient of.
        obs = column name of actual values (number of claims) '''
    
    d1 = gini_table(df, pred, act)
    Gini_coef = round((d1.sum()['gini_area'] - 0.5) *2,6)
    return(Gini_coef)