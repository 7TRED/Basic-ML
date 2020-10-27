from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np


# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data = boston_dataset.data, columns = boston_dataset.feature_names)
features = data.drop(['INDUS','AGE'],axis = 1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns = ['PRICE'])


CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
PTRATIO_IDX = 8
RM_IDX = 4


# property_stats[0][CRIME_IDX] = features['CRIM'].mean()
# property_stats[0][ZN_IDX] = features['ZN'].mean()
# property_stats[0][CHAS_IDX] = features['CHAS'].mean()
# property_stats[0][RM_IDX] = features['RM'].mean()

property_stats = features.mean().values.reshape(1,11)

regr = LinearRegression()
regr.fit(features, target)

fitted_vals = regr.predict(features)
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


def getLogEstimate(roomNO, studentPerClass, nxtToRiver=False, highConfid=True ):
    #Configure Property
    property_stats[0][RM_IDX] = roomNO
    property_stats[0][PTRATIO_IDX] = studentPerClass
    
    if nxtToRiver:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    # Make Predictions
    log_estimate = regr.predict(property_stats)[0][0]
    
    # Calculate Range
    if highConfid:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 68
    return log_estimate, upper_bound, lower_bound, interval


def getDollarEstimate( rm, ptratio, chas=False, large_range=True ):
    
    """
        Estimate the price of a property in boston.
        
        PARAMETERS:
        
        rm -- number of rooms in the property
        ptratio -- number of students per teacher in the classroom for the school
        chas -- True if the property is next to the river, False otherwise.
        large_range -- True for 95% prediction interval, False for a 68% interval.
    
    """
    ZILLOW_MEDIAN_PRICE = 583.3
    factor = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)
    
    if rm < 1 or ptratio < 1 :
        print("Unrealistic No. of rooms")
        return
    
    log_est, upper, lower, conf = getLogEstimate(rm, studentPerClass = ptratio, 
                                                 nxtToRiver = chas, 
                                                 highConfid= large_range)
    # Convert to today's Dollars 
    dollar_est = np.e**log_est * 1000 * factor
    dollar_hi = np.e**upper * 1000 * factor
    dollar_low = np.e**lower * 1000 * factor

    # Round the dollar values to nearest thousand
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f"Estimated Price : ${rounded_est}")
    print(f"Range : ${rounded_low} - ${rounded_hi}\nConfidence : {conf}")


