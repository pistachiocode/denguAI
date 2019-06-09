#!/usr/bin/env python
# coding: utf-8

# DengAI Competition
 
# RStudio configuration

library(reticulate)
reticulate::use_python('/anaconda3/bin')

reticulate::repl_python()

# Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dengue_utils as dutils
import dsutils as du

# -------------------
# Init
# -------------------

os.chdir("/Users/rocio/Dropbox/DataScience/dengAI/scripts")

features_train = pd.read_csv(dutils.FEATURES_TRAIN_PATH)
labels_train = pd.read_csv(dutils.LABELS_TRAIN_PATH)

features_test = pd.read_csv(dutils.FEATURES_TEST_PATH)


# Join of train features and labels features

train_dataset = features_train.merge(labels_train, 
                                    left_on=['city', 'year', 'weekofyear'], 
                                    right_on=['city', 'year', 'weekofyear'],
                                    how='inner')


# -----------------------
# 1. Boxplots all features
# -----------------------

du.show_boxplot(train_dataset, exclude=['year'])
plt.savefig(dutils.GRAPH_PATH + '/01.01_boxplot_allfeatures.png', bbox_inches="tight")

# Conversion kelvin to celsius
train_dataset = dutils.temperature_conversion(train_dataset)

du.show_boxplot(train_dataset, exclude=['year'])
plt.savefig(dutils.GRAPH_PATH + '/01.02_boxplot_allfeatures_celcius.png', bbox_inches="tight")

# -----------------------
# 2. Heatmap - correlation
# -----------------------

# Correlation matrix
du.show_heatmap(train_dataset, exclude = ['city'])
plt.savefig(dutils.GRAPH_PATH + '/01.03_heatmap_correlations.png', bbox_inches="tight")

# -----------------------
# 3. Line plots
# -----------------------

train_dataset['week_start_date'] = pd.to_datetime(train_dataset['week_start_date'])

du.show_lineplot(train_dataset, xvalue='week_start_date', yvalue='total_cases', hue='city')
plt.savefig(dutils.GRAPH_PATH + '/01.04_lineplots_totalcases.png', bbox_inches="tight")

du.show_lineplot(train_dataset, xvalue='week_start_date', yvalue='reanalysis_avg_temp_c', hue='city')
plt.savefig(dutils.GRAPH_PATH + '/01.04_lineplots_reanalysis_avg_temp_c.png', bbox_inches="tight")

du.show_lineplot(train_dataset, xvalue='week_start_date', yvalue='reanalysis_relative_humidity_percent', hue='city')
plt.savefig(dutils.GRAPH_PATH + '/01.04_lineplots_reanalysis_relative_humidity_percent.png', bbox_inches="tight")

du.show_lineplot(train_dataset, xvalue='week_start_date', yvalue='precipitation_amt_mm', hue='city')
plt.savefig(dutils.GRAPH_PATH + '/01.04_lineplots_precipitation_amt_mm.png', bbox_inches="tight")


# -----------------------
# Density plots
# -----------------------

# features = train_dataset.columns.drop(['city', 'week_start_date'])
# 
# fig = plt.figure(figsize=(50, 50))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# 
# for i, column in enumerate(features):
# 
#     ax = fig.add_subplot(5, 5, i+1)
# 
#     data_imputed = train_dataset[column]
#     
#     sns.distplot(data_imputed, hist=False, kde=True, bins=int(180/5), color = 'salmon', 
#              hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 3})
#     
#     ax.axes.set_title(column,fontsize=16)
#     ax.set_xlabel(column, fontsize=12)
# 
# plt.savefig(dutils.GRAPH_PATH + '/01.06_density_features.png', bbox_inches="tight")

# -----------------------
# 4. Scatter plots
# -----------------------

du.show_scatterplot_matrix(train_dataset, 
                           y='total_cases', 
                           ylabel = "Total Sases", 
                           exclude = ['city', 'total_cases','week_start_date'])    

plt.savefig(dutils.GRAPH_PATH + '/01.05_scatterplots_label_vs_features.png', bbox_inches="tight")


for city in  train_dataset['city'].unique():
  
  train_dataset_city = train_dataset[train_dataset['city'] == city]
  
  du.show_scatterplot_matrix(train_dataset_city, 
                             y='total_cases', 
                             ylabel = "Total Sases", 
                             exclude = ['city', 'total_cases','week_start_date'])    
  
  plt.savefig(dutils.GRAPH_PATH + '/01.05_scatterplots_label_vs_features_{}.png'.format(city), bbox_inches="tight")

# -----------------------------
# 5. Features correlation barplot
# ----------------------------

for city in train_dataset['city'].unique():
  
  train_dataset_city = train_dataset[train_dataset['city'] == city]
  
  du.show_feature_correlation(train_dataset_city, 'total_cases', 'Features correlations city {}'.format(city))
  
  plt.savefig(dutils.GRAPH_PATH + '/01.06_barplot_feature_vs_label_corr_{}.png'.format(city), bbox_inches="tight")


# ## Add month of the year
# 
# train_dataset['week_start_date'] = pd.to_datetime(train_dataset['week_start_date'])
# train_dataset['monthofyear'] = train_dataset['week_start_date'].apply(lambda x: x.month)
# 
# # High correlation between ndvi_nw-ndvi-ne and ndvi-sw-ndvi-se
# # Add the mean of each pair that indicates the level of vegetation in the north and south of both cities.
# 
# # Features engineering
# 
# train_dataset['ndvi_north'] = train_dataset[['ndvi_nw', 'ndvi_ne']].mean(axis=1)
# train_dataset['ndvi_south'] = train_dataset[['ndvi_sw', 'ndvi_se']].mean(axis=1)
# 
# #Remove feature
# 
# train_dataset = train_dataset.drop(['ndvi_sw', 'ndvi_nw', 'ndvi_ne', 'ndvi_se', 'week_start_date'], axis=1)
# 
# 
# du.show_scatterplot_matrix(train_dataset, 
#                            y='total_cases', 
#                            ylabel = "Total Cases", 
#                            exclude = ['city', 'total_cases'])    
# 


##features = train_dataset.columns.drop(['week_start_date'])
#
#fig = plt.figure(figsize=(20, 20))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#
#for i, column in enumerate(features):
#
#    ax = fig.add_subplot(5, 5, i+1)
#
#    data_sj = train_dataset[train_dataset['city']=='sj'][column]
#    data_iq = train_dataset[train_dataset['city']=='iq'][column]
#    
#    sns.distplot(data_sj, hist=False, kde=True, bins=int(180/5), color = 'salmon', 
#             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 3})
#    
#    sns.distplot(data_iq, hist=False, kde=True, bins=int(180/5), color = 'darkblue', 
#             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 3})




