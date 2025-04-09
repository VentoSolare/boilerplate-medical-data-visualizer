import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("medical_examination.csv")


bmi = df['weight'] / (df['height'] / 100)**2
df['overweight'] = (bmi > 25).astype(int)


df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    df_cat = df[['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight', 'cardio']]

    
    df_cat = pd.melt(df_cat, id_vars='cardio', var_name='variable', value_name='value')

    
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).size()
    df_cat.rename(columns={"size": "total"}, inplace=True)

    
    fig = sns.catplot(
        x="variable",      
        y="total",           
        hue="value",         
        col="cardio",        
        data=df_cat,         
        kind="bar",          
        height=5,            
        aspect=1.5           
    )

    
    fig.savefig('catplot.png')
    return fig



def draw_heat_map():
    
    df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
]


    corr = df_heat.corr(numeric_only = True)


    mask = np.triu(np.ones_like(corr, dtype = bool))


    fig, ax = plt.subplots(figsize=(12, 10))


    sns.heatmap( 
    corr,
    mask=mask,
    annot=True,
    fmt=".1f",
    center=0,
    vmin=-0.15,
    vmax=0.3,
    square=True,
    linewidths=0.5,
    cbar_kws={
        "shrink": 0.5,
        "ticks": [-0.08, 0.00, 0.08, 0.16, 0.24]
    },
    ax=ax
    )

    fig.savefig('heatmap.png')
    return fig
