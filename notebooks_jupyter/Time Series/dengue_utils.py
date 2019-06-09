import pandas as pd
import numpy as np
import pytemperature

from datetime import datetime

# ------------------------------
# Constants
# ------------------------------

GRAPH_PATH = '../results/plots'
SUBMISIONS_PATH = '../results/submisions'

FEATURES_TRAIN_PATH = '../data/dengue_features_train.csv'
LABELS_TRAIN_PATH = '../data/dengue_labels_train.csv'
FEATURES_TEST_PATH = '../data/dengue_features_test.csv'


# ------------------------------
#  Load data
# ------------------------------

def load_train_dataset():
    """
     Load test and train datasets as pandas dataframes
    """
    
    printlog("Loading train dataset from: " + FEATURES_TRAIN_PATH)
    
    features_train = pd.read_csv(FEATURES_TRAIN_PATH)
    labels_train = pd.read_csv(LABELS_TRAIN_PATH)

    train_dataset = features_train.merge(labels_train,
                                         left_on=['city', 'year', 'weekofyear'], 
                                         right_on=['city', 'year', 'weekofyear'],
                                         how='inner')
    
    return train_dataset

def load_test_dataset(): 
    """
     Load test datasets as pandas dataframes
    """
    printlog("Loading train dataset from: " + FEATURES_TEST_PATH)
    
    test_dataset = pd.read_csv(FEATURES_TEST_PATH)
    
    return test_dataset


# ------------------------------
#  Features engineering
# ------------------------------


def temperature_conversion_features(data, feature, new_feature):
    
    if feature in data.columns:
        printlog("temperature conversion: kelvin to celsius " + feature + " to " + new_feature)
        data[new_feature] = pytemperature.k2c(data[feature])


def temperature_conversion(data):
    """
        This method converts some features to kelvin to celsius.
    """
    
    features = ['reanalysis_air_temp_k', 'reanalysis_dew_point_temp_k', 
                'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                'reanalysis_avg_temp_k']
    
    new_features = ['reanalysis_air_temp_c', 'reanalysis_dew_point_temp_c', 
                    'reanalysis_max_air_temp_c', 'reanalysis_min_air_temp_c',
                    'reanalysis_avg_temp_c']
    
    
    for i,feature in enumerate(features):
        datau = temperature_conversion_features(data, feature, new_features[i])
        
    data = data.drop(features, axis=1)
    
    return (data)


def remove_outliers_iqr(data, feature, exclude = list()):
    """
        Remove outliers of a feature using IQR method. This method returns
        a pandas Series with nan values instead of outliers.
        
        Only numeric features are processed. Features included in
        exclude list are ignored.
    
    """
    
    if np.issubdtype(data[feature].dtype, np.number) and feature not in exclude:
        
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)

        iqr = q3 - q1
        lower_bound = q1 -(1.5 * iqr) 
        upper_bound = q3 +(1.5 * iqr) 

        outliers = data[feature].apply(lambda x : np.nan if (x < lower_bound or 
                                                             x > upper_bound) 
                                       else x)
    else:
        outliers = data[feature]
        
    return outliers

def get_segment_precipitation(sample):
    
    # precipitations
    
    if (sample['reanalysis_precip_amt_kg_per_m2'] > 50 or
        sample['reanalysis_sat_precip_amt_mm'] > 75):
        return 1
    else:
        return 0


def get_segment_humidity(sample):
    
    # humidity
    
    if (sample['reanalysis_relative_humidity_percent'] > 90 or
        sample['reanalysis_specific_humidity_g_per_kg'] > 18):
        return 1
    else:
        return 0
    
        
def get_segment_temperature(sample):
    # temperature
    
    if (sample['reanalysis_avg_temp_c'] > 27.3):
        return 1
    else:
        return 0

def get_segment(sample):
    
    segment_precipitation = ""
    segment_humidity = ""
    segment_temperature = ""
    
    # precipitations
    
    if (sample['reanalysis_precip_amt_kg_per_m2'] > 58 or
        sample['reanalysis_sat_precip_amt_mm'] > 84):
        segment_precipitation = "HP"
    elif (sample['reanalysis_precip_amt_kg_per_m2'] < 10 or
             sample['reanalysis_sat_precip_amt_mm'] < 10):
        segment_precipitation = "LP"
    else:
        segment_precipitation = "NP"
        
        
    # humidity
    
    if (sample['reanalysis_relative_humidity_percent'] > 92 or
        sample['reanalysis_specific_humidity_g_per_kg'] > 18):
        segment_humidity = "HH"
    elif (sample['reanalysis_relative_humidity_percent'] < 75 or
             sample['reanalysis_specific_humidity_g_per_kg'] < 15):
        segment_humidity = "LH"
    else:
        segment_humidity = "NH"
        
    # temperature
    
    if (sample['station_avg_temp_c'] > 27.5):
        segment_temperature = "HT"
    elif (sample['station_avg_temp_c'] < 25):
        segment_temperature = "LT"
    else:
        segment_temperature = "NT"
    
    segment = segment_precipitation + segment_humidity + segment_temperature
    return segment
    
    
# ------------------------------
#  Utils
# ------------------------------

def printlog(text):
    """
        Print log with date and time
    """

    print(datetime.today().strftime("%Y%m%d - %H%:M:%S") + ": " + text)

