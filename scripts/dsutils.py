import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime

sns.set_style("whitegrid")

from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score

def is_numeric_feature(df, feature):
    """
        This functions reaturn a boolean value that indicates if a column
        in a input dataframe is numeric.
    """
    return np.issubdtype(df[feature].dtype, np.number) 


def get_metrics(y_test, y_pred):
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ex_var = explained_variance_score(y_test, y_pred)
    
    return mae, r2, ex_var


# ------------------------------
#  Models
# ------------------------------

def simple_linear_regression(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=83)
    
    reg = LinearRegression(normalize=True).fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ex_var = explained_variance_score(y_test, y_pred)
    
    return mae, r2, ex_var


def enet(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=83)

    reg = ElasticNet(normalize=False).fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    
    return y_pred, y_test




# ------------------------------
#  Plots
# ------------------------------

def show_boxplot(df, features=None, exclude=[], figsize=(12, 15)):
    
    features_boxplot = []
    if features==None:
        features = df.columns
    
    for column in features:
        if column not in exclude and is_numeric_feature(df, column):
            features_boxplot.append(column)
        else:
            printlog("show_bloxplot: omiting " + column)

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=figsize)
    
    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=df[features_boxplot], whis="range", palette="husl", orient="h")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)


    
def show_lineplot(df, xvalue, yvalue, hue=None, figsize=(15, 10)):
    f, ax = plt.subplots(figsize=(15, 10))

    # Plot the responses for different events and regions
    sns.lineplot(x=xvalue, y=yvalue, hue=hue, data=df)
    
def show_heatmap(df, exclude=[]):
    
    plt.figure(figsize=(18,18))

    font = {'size'   : 8}

    plt.rc('font', **font)

    for c in exclude:
        df_corr = df.drop(c, axis=1).corr()
    
    ax = sns.heatmap(df_corr, annot=True, cmap="coolwarm")

    
def show_feature_correlation(df, label, title = "", exclude=[]):
    
    fig = plt.figure(figsize=(8, 10))
    font = {'size'   : 14}

    plt.rc('font', **font)

    df = df.drop(exclude, axis=1)
    df_corr = df.corr()
    df_corr= df_corr.sort_values(label)[[label]]

    values = df_corr.values
    clrs = ['grey' if (np.abs(x) < 0.3) else '#A5DF00' for x in values ]
    ax = sns.barplot(y=df_corr.index, x=label, data=df_corr, palette=clrs)
    ax.axes.set_title(title, fontsize=30)

def show_scatterplot_matrix(df, y, ylabel, exclude=[]):
    
    features = df.columns.drop(exclude)

    fig = plt.figure(figsize=(60, 80))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, column in enumerate(features):

        ax = fig.add_subplot(5, 5, i+1)

        dot_color = 'salmon'
        if "temp" in column:
            dot_color = 'c'

        sns.scatterplot(x=column, y=y, data=df, color=dot_color)
        ax.axes.set_title(column,fontsize=16)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
# ------------------------------
#  Utils
# ------------------------------

def printlog(text):
    """
        Print log with date and time
    """

    print(datetime.today().strftime("%Y%m%d - %H%:M:%S") + ": " + text)

