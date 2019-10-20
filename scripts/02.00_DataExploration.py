#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:48:09 2019

@author: rocio
"""

import os
os.chdir("/Users/rocio/Dropbox/DataScience/dengAI/scripts")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dengue_utils as dutils
import dsutils as du
import pytemperature 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

SAVE_PLOTS = True

# -------------------
# Init
# -------------------



features_train = pd.read_csv(dutils.FEATURES_TRAIN_PATH)
labels_train   = pd.read_csv(dutils.LABELS_TRAIN_PATH)

features_test = pd.read_csv(dutils.FEATURES_TEST_PATH)


# Join of train features and labels features

train_dataset = features_train.merge(labels_train, 
                                    left_on=['city', 'year', 'weekofyear'], 
                                    right_on=['city', 'year', 'weekofyear'],
                                    how='inner')


# -----------------------
# 1. Missing values
# -----------------------





# -----------------------
# 1. Line plots
# -----------------------

train_dataset['week_start_date'] = pd.to_datetime(train_dataset['week_start_date'])

train_dataset_sj = train_dataset.loc[train_dataset['city']=='sj',:]

du.show_lineplot(train_dataset_2007, xvalue='weekofyear', yvalue='total_cases', hue='year')


train_dataset_iq = train_dataset.loc[train_dataset['city']=='iq',:]

du.show_lineplot(train_dataset_2007, xvalue='weekofyear', yvalue='total_cases', hue='year')



























