# operating system functions
from numpy import array, exp
from pandas import *
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import sklearn.model_selection
import sklearn.preprocessing
import sklearn
import matplotlib.pyplot as plt
import seaborn as sbn
import scipy
import numpy as np
import pandas as pd
import os
import time
import re
import random
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# data science, linear algebra functions

# data visualization

# machine learning


def remove_non_numeric(x):
    if x == x:
        return re.sub('[^0-9.-]', '', str(x))
    else:
        return 0


nrowsConst = 300000
rowsToSkip = 0
#read in data
# set file directory
os.chdir('C://Users//Student8//Documents//files_to_transfer//files_to_transfer/')
product_data = pd.read_csv(
    'product_data.csv', skiprows=rowsToSkip, nrows=nrowsConst)
trans_data = pd.read_csv('transaction_data.csv',
                         skiprows=rowsToSkip, nrows=nrowsConst)
cust_data = pd.read_csv('customer_data.csv', usecols=['mastercustomerid', 'zipcode', 'c_ethnicpoppcblackhisp', 'c_buyerloyaltycarduser', 'c_commutewrkrspccarpooled', 'c_employpop18pluspccivilvet', 'c_edupop25plusmededucattain', 'c_educpop25pluspcgradprofdeg', 'c_actintvideogamer', 'c_actintnflenthusiast', 'c_actintmlbenthusiast', 'c_actintdogowners', 'c_actintdoityourselfers', 'c_person1technologyadoption', 'c_oldervsnewerbuilthomessco', 'c_healthfitnessmagbyrcat', 'c_estimatedcurrenthomevalue', 'c_actintplaygolf', 'c_buyeryoungadultclothshop', 'c_builthupcbuilt20002004', 'c_activeoutdoorsscore', 'age', 'c_retailelectrongadg',
                                                      'c_actintcountrymusic', 'c_actintfitnessenthusiast', 'c_actinthealthyliving', 'c_actintoutdoorenthusiast', 'c_actintsportsenthusiast', 'c_actintpgatourenthusiast', 'c_educationindividual1', 'c_hhconsumerexpendituresshoes', 'c_newvsused', 'c_hhconsumerexpendtravel', 'c_menscasualappareldollar', 'c_buyernondeptstoremakeup', 'c_workingcouples', 'c_healthimage', 'c_buyerhighendspiritdrinker', 'c_actintsportsenthusiast', 'c_fincorporatecreditcardusr', 'c_educpop25pluspchsdiploma', 'c_buyerluxurystoreshop', 'c_ethnichhpchohwhitenhsp', 'c_esthhincomeamountv5', 'c_retailtravel', 'c_brandloyal'], skiprows=rowsToSkip, nrows=nrowsConst)

print("loading done")

# to eliminate extra values in mastercustomerid and prep for trans_data join with customer data
bool_mask = [x.startswith('FTP') for x in trans_data['mastercustomerid']]
trans_data = trans_data[bool_mask]
trans_data['mastercustomerid'] = [x[14:]
                                  for x in trans_data['mastercustomerid'] if x.startswith('FTP')]

# create a smaller transaction file, with only meaningful mastercustomerid values (recommended)
trans_data.to_csv('smaller_transaction_data.csv', index=False)

# merge customer transactions with customer demographics: caution, memory intensive
cust_trans_data2 = trans_data.merge(
    cust_data, how='inner', left_on='mastercustomerid', right_on='mastercustomerid')

# use 'groupby' to answer questions about
# how much did each customer spend on average?

cust_trans_data = cust_trans_data2.merge(
    product_data, how='inner', left_on='productid', right_on='productid')

# adding colors
cust_trans_data['is_black'] = [
    1 if x == 'BLACK' else 0 for x in cust_trans_data['color']]
cust_trans_data['is_blue'] = [1 if (x == 'BLUE' or x == 'LIGHT BLUE' or x == 'BRIGHT BLUE' or x ==
                                    'TEAL' or x == 'DARK BLUE' or x == "TURQUOISE") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_navy'] = [1 if (
    x == 'NAVY' or x == 'BRIGHT NAVY' or x == 'DARK NAVY') else 0 for x in cust_trans_data['color']]
cust_trans_data['is_grey'] = [1 if (x == 'GREY' or x == 'LIGHT GREY' or x == 'DARK GREY' or x ==
                                    "CAMBRIDGE GREY" or x == "CHARCOAL" or x == 'STONE') else 0 for x in cust_trans_data['color']]
cust_trans_data['is_tan'] = [1 if (x == 'TAN' or x == "BRITISH TAN" or x == "LIGHT TAN" or x ==
                                   "DARK TAN" or x == "CAMEL") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_brown'] = [1 if (x == 'BROWN' or x == 'DARK BROWN' or x == 'TAUPE' or x ==
                                     'BROWN' or x == "LIGHT BROWN" or x == "MOCHA") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_no_color'] = [
    1 if x == 'NO COLOR' else 0 for x in cust_trans_data['color']]
cust_trans_data['is_purple'] = [1 if (x == 'PURPLE' or x == 'LIGHT PURPLE' or x ==
                                      'DARK PURPLE' or x == "WINE") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_burgundy'] = [
    1 if x == 'BURGUNDY' else 0 for x in cust_trans_data['color']]
cust_trans_data['is_white'] = [1 if (x == 'WHITE' or x == 'NATURAL' or x == 'CREAM' or x ==
                                     'OFF WHITE' or x == "IVORY") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_olive'] = [1 if (
    x == 'OLIVE' or x == 'LIGHT OLIVE' or x == 'DARK OLIVE') else 0 for x in cust_trans_data['color']]
cust_trans_data['is_red'] = [1 if (x == 'RED' or x == 'RUST' or x == 'LIGHT RED' or x ==
                                   "BRIGHT RED" or x == "DARK RED") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_blackwhite'] = [
    1 if x == 'BLACK/WHITE' else 0 for x in cust_trans_data['color']]
cust_trans_data['is_pink'] = [
    1 if (x == 'PINK' or x == 'LIGHT PINK') else 0 for x in cust_trans_data['color']]
cust_trans_data['is_green'] = [1 if (x == 'GREEN' or x == 'LIGHT GREEN' or x ==
                                     'BRIGHT GREEN' or x == "DARK GREEN") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_yellow'] = [1 if (x == 'YELLOW' or x == 'GOLD' or x == 'LIGHT YELLOW' or x == 'BRIGHT YELLOW' or x == "METAL GOLD" or x == "MUSTARD" or x ==
                                      "DARK GOLD" or x == "METAL BRONZE" or x == "DARK YELLOW" or x == "METAL BRASS" or x == "BRONZE") else 0 for x in cust_trans_data['color']]
cust_trans_data['is_orange'] = [1 if (x == 'ORANGE' or x == 'LIGHT ORANGE' or x ==
                                      'DARK ORANGE' or x == 'BRIGHT ORANGE') else 0 for x in cust_trans_data['color']]

# adding brands
cust_trans_data['is_josABank'] = [
    1 if x == 'J. BANK MFG.' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_colehaan'] = [
    1 if x == 'COLE HAAN' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_allenedmonds'] = [
    1 if x == 'ALLEN EDMONDS' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_johnston'] = [
    1 if x == 'JOHNSTON & MURPHY' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_misc'] = [
    1 if x == 'MISC VENDOR' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_jaapparel'] = [
    1 if x == 'J A APPAREL CORP/ GMAC' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_redcollar'] = [
    1 if x == 'RED COLLAR QINGDAO' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_monzini'] = [
    1 if x == 'CONFECCIONES MONZINI' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_dayang'] = [
    1 if x == 'DAYANG DALIAN MODA FASH' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_rockport'] = [
    1 if x == 'ROCKPORT' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_unified'] = [
    1 if x == 'UNIFIED ACCESSORIES' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_greatchina'] = [
    1 if x == 'GREAT CHINA EMPIRE' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_ghbass'] = [
    1 if x == 'G. H. BASS' else 0 for x in cust_trans_data['c_vendorname']]
cust_trans_data['is_lbevans'] = [
    1 if x == 'L. B. EVANS' else 0 for x in cust_trans_data['c_vendorname']]

# adding items
cust_trans_data['is_suit'] = [
    1 if x == 'SUITS' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_shoes'] = [
    1 if x == 'SHOES' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_dressshirt'] = [
    1 if x == 'DRESS SHIRTS' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_slacks'] = [
    1 if x == 'SLACKS' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_sportcoat'] = [
    1 if x == 'SPORT COATS' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_sportswear'] = [
    1 if x == 'SPORTSWEAR' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_accesories'] = [
    1 if x == 'ACCESSORIES' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_outerwear'] = [
    1 if x == 'OUTER WEAR' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_services'] = [
    1 if x == 'SERVICES' else 0 for x in cust_trans_data['c_divisionname']]
cust_trans_data['is_null'] = [
    1 if x == 'NULL' else 0 for x in cust_trans_data['c_divisionname']]


def convertItem(s):
    arr1 = ['is_suit', 'is_shoes', 'is_dressshirt', 'is_slacks', 'is_sportcoat',
            'is_sportswear', 'is_accesories', 'is_outerwear', 'is_services', 'is_null']
    arr2 = ['SUITS', 'SHOES', 'DRESS SHIRTS', 'SLACKS', 'SPORT COATS',
            'SPORTSWEAR', 'ACCESSORIES', 'OUTER WEAR', 'SERVICES', 'NULL']
    return arr1[arr2.index(s)]


def convertColor(s):
    arr1 = ['is_black', 'is_blue', 'is_navy', 'is_grey', 'is_tan', 'is_brown', 'is_no_color', 'is_purple',
            'is_burgundy', 'is_white', 'is_olive', 'is_red', 'is_blackwhite', 'is_pink', 'is_green', 'is_yellow', 'is_orange']
    arr2 = [['BLACK'], ['BLUE', 'LIGHT BLUE', 'BRIGHT BLUE', 'TEAL', 'DARK BLUE', 'TURQUOISE'], ['NAVY', 'BRIGHT NAVY', 'DARK NAVY'], ['GREY', 'LIGHT GREY', 'DARK GREY', 'CAMBRIDGE GREY', 'CHARCOAL', 'STONE'], ['TAN', "BRITISH TAN", "LIGHT TAN", "DARK TAN", "CAMEL"], ['BROWN', 'DARK BROWN', 'TAUPE', 'BROWN', "LIGHT BROWN", "MOCHA"], ['NO COLOR'], ['PURPLE', 'LIGHT PURPLE', 'DARK PURPLE', "WINE"], ['BURGUNDY'], ['WHITE', 'NATURAL',
                                                                                                                                                                                                                                                                                                                                                                                                                               'CREAM', 'OFF WHITE', "IVORY"], ['OLIVE', 'LIGHT OLIVE', 'DARK OLIVE'], ['RED', 'RUST', 'LIGHT RED', "BRIGHT RED", "DARK RED"], ['BLACK/WHITE'], ['PINK', 'LIGHT PINK'], ['GREEN', 'LIGHT GREEN', 'BRIGHT GREEN', "DARK GREEN"], ['YELLOW', 'GOLD', 'LIGHT YELLOW', 'BRIGHT YELLOW', "METAL GOLD", "MUSTARD", "DARK GOLD", "METAL BRONZE", "DARK YELLOW", "METAL BRASS", "BRONZE"], ['ORANGE', 'LIGHT ORANGE', 'DARK ORANGE', 'BRIGHT ORANGE']]
    for i in range(len(arr2)):
        for j in range(len(arr2[i])):
            if (arr2[i][j] == s):
                return arr1[i]


cust_trans_data.groupby(by='mastercustomerid')['is_black'].mean()

# how much did each customer spend in total?
cust_trans_data.groupby(by='mastercustomerid')['is_black'].sum()

# list of products each customer purchased
cust_trans_data.groupby(by='mastercustomerid')['productid'].apply(list)

# Cross validated model construction

# choose the sklearn model you want to use, a few are suggested but hundreds are available
model = LogisticRegression(max_iter=10000)
# model = LinearRegression()
# model=RandomForestClassifier(n_estimators=50, oob_score=True)
# model=GradientBoostingRegressor(learning_rate=.01)

# select the columns you want to use to help make your prediction.
col_to_use_for_pred = ['age', 'c_activeoutdoorsscore', 'c_actintplaygolf', 'c_ethnicpoppcblackhisp', 'c_buyerloyaltycarduser', 'c_commutewrkrspccarpooled', 'c_employpop18pluspccivilvet', 'c_edupop25plusmededucattain', 'c_educpop25pluspcgradprofdeg', 'c_actintvideogamer', 'c_actintnflenthusiast', 'c_actintmlbenthusiast', 'c_actintdogowners', 'c_actintdoityourselfers', 'c_person1technologyadoption', 'c_oldervsnewerbuilthomessco', 'c_healthfitnessmagbyrcat', 'c_estimatedcurrenthomevalue', 'c_buyeryoungadultclothshop', 'c_builthupcbuilt20002004', 'c_retailelectrongadg',
                       'c_actintcountrymusic', 'c_actintfitnessenthusiast', 'c_actinthealthyliving', 'c_actintoutdoorenthusiast', 'c_actintsportsenthusiast', 'c_actintpgatourenthusiast', 'c_educationindividual1', 'c_hhconsumerexpendituresshoes', 'c_newvsused', 'c_healthimage', 'c_workingcouples', 'c_hhconsumerexpendtravel', 'c_menscasualappareldollar', 'c_buyernondeptstoremakeup', 'c_ethnichhpchohwhitenhsp', 'c_buyerhighendspiritdrinker', 'c_actintsportsenthusiast', 'c_fincorporatecreditcardusr', 'c_educpop25pluspchsdiploma', 'c_buyerluxurystoreshop', 'c_esthhincomeamountv5', 'c_brandloyal', 'c_retailtravel']
targets = []
# target_col = ['is_black','is_blue', 'is_navy', 'is_grey', 'is_tan', 'is_brown', 'is_no_color', 'is_purple', 'is_burgundy', 'is_white', 'is_olive', 'is_red', 'is_blackwhite', 'is_pink', 'is_green', 'is_yellow', 'is_orange']
color_target_col = ['is_black', 'is_blue', 'is_navy', 'is_grey', 'is_tan', 'is_brown', 'is_no_color', 'is_purple',
                    'is_burgundy', 'is_white', 'is_olive', 'is_red', 'is_blackwhite', 'is_pink', 'is_green', 'is_yellow', 'is_orange']
brand_target_col = ['is_josABank', 'is_colehaan', 'is_allenedmonds', 'is_johnston', 'is_misc', 'is_jaapparel',
                    'is_redcollar', 'is_monzini', 'is_dayang', 'is_rockport', 'is_unified', 'is_greatchina', 'is_ghbass', 'is_lbevans']
item_target_col = ['is_suit', 'is_shoes', 'is_dressshirt', 'is_slacks',
                   'is_sportcoat', 'is_sportswear', 'is_accesories', 'is_outerwear', 'is_services']
targets.append(color_target_col)
# targets.append(brand_target_col)
targets.append(item_target_col)

for target_col in targets:
    # get rid of any nans in target column
    cust_trans_data = cust_trans_data.dropna(subset=target_col)

    # you'll have to figure out how to deal with nans in your data sources. You can set them to zero (linear models), the median
    # value (tree-based model), or drop them entirely

    cust_trans_data = cust_trans_data.fillna(0)  # replace with zero
    # for x in col_to_use_for_pred:
    #   cust_trans_data[x] = cust_trans_data[x].fillna(cust_trans_data[x].median())  # replace with median
    # cust_trans_data = cust_trans_data.dropna() #drop na values


final_weights = []

splitLine = (int)(len(cust_trans_data)*0.90)
print('splitLine ' + str(splitLine))
df1 = cust_trans_data.iloc[:splitLine, :]
df2 = cust_trans_data.iloc[splitLine:, :]

cust_trans_data = df1
test_data = []
test_data = df2

for target_col in targets:
    weights = []

    # x = []
    # for i in range(0, len(test[col_to_use_for_pred].keys())): x.append(test[col_to_use_for_pred].keys()[i])
    # final_weights.append(x)

    for color in target_col:

        mean_error = []
        total_error = []
        mean_values = []

        # this section of the code is responsible for cross-validation. Set the number of chunks you want to break your data into, and the model will train/test based on each of the chunks.
        temp = list(range(5))
        for i in list(range(5)):  # iterate over the 5 cross validation segments

            # this section breaks up the dataset into a training and test dataset
            temp = list(range(5))
            temp.remove(i)
            test = cust_trans_data[i::5]
            training_setup = [cust_trans_data[temp[0]::5], cust_trans_data[temp[1]::5], cust_trans_data[temp[2]::5],
                              cust_trans_data[temp[3]::5]]
            training = pd.concat(training_setup)

            fit = model.fit(training[col_to_use_for_pred],
                            training[color])  # fit the model
            prediction = fit.predict(
                test[col_to_use_for_pred])  # make the prediction

            mean_error.append(np.mean([np.abs(x-y)
                                       for x, y in zip(prediction, test[color])]))

            total_error.append(np.abs([np.abs(x-y)
                                       for x, y in zip(prediction, test[color])]))

            mean_values.append(np.abs([np.abs(x-y)
                                       for x, y in zip(prediction, test[color])]))

        # print(mean_error)
        # print(np.mean(mean_error))
        # print('Feature Weights')
        # print(color)

        # this code will tell you which columns were significant in making the prediction. One works for linear regression, the other
        # for decision tree types of models
        # for i in range(0, len(test[col_to_use_for_pred].keys())): print(fit.coef_[0][i], test[col_to_use_for_pred].keys()[i])
        output = []
        for i in range(0, len(test[col_to_use_for_pred].keys())):
            output.append(fit.coef_[0][i])
        weights.append(output)

        #for i in range(0, len(test[col_to_use_for_pred].keys())): print(fit.feature_importances_[i], test[col_to_use_for_pred].keys()[i])
        #output = []
        #for i in range(0, len(test[col_to_use_for_pred].keys())): output.append(fit.feature_importances_[i])
        # weights.append(output)

    final_weights.append(weights)

print(final_weights)

colors = []
items = []

#testing_data = cust_trans_data[['age', 'c_activeoutdoorsscore', 'c_actintplaygolf', 'c_ethnicpoppcblackhisp', 'c_buyerloyaltycarduser', 'c_commutewrkrspccarpooled', 'c_employpop18pluspccivilvet', 'c_edupop25plusmededucattain', 'c_educpop25pluspcgradprofdeg', 'c_actintvideogamer', 'c_actintnflenthusiast', 'c_actintmlbenthusiast', 'c_actintdogowners','c_actintdoityourselfers', 'c_person1technologyadoption', 'c_oldervsnewerbuilthomessco', 'c_healthfitnessmagbyrcat', 'c_estimatedcurrenthomevalue', 'c_buyeryoungadultclothshop', 'c_builthupcbuilt20002004', 'c_retailelectrongadg','c_actintcountrymusic','c_actintfitnessenthusiast', 'c_actinthealthyliving', 'c_actintoutdoorenthusiast' ,'c_actintsportsenthusiast' ,'c_actintpgatourenthusiast', 'c_educationindividual1', 'c_hhconsumerexpendituresshoes', 'c_newvsused', 'c_healthimage','c_workingcouples','c_hhconsumerexpendtravel','c_menscasualappareldollar','c_buyernondeptstoremakeup','c_ethnichhpchohwhitenhsp', 'c_buyerhighendspiritdrinker', 'c_actintsportsenthusiast', 'c_fincorporatecreditcardusr', 'c_educpop25pluspchsdiploma', 'c_buyerluxurystoreshop', 'c_esthhincomeamountv5','c_brandloyal', 'c_retailtravel']]
testing_data = test_data[['age', 'c_activeoutdoorsscore', 'c_actintplaygolf', 'c_ethnicpoppcblackhisp', 'c_buyerloyaltycarduser', 'c_commutewrkrspccarpooled', 'c_employpop18pluspccivilvet', 'c_edupop25plusmededucattain', 'c_educpop25pluspcgradprofdeg', 'c_actintvideogamer', 'c_actintnflenthusiast', 'c_actintmlbenthusiast', 'c_actintdogowners', 'c_actintdoityourselfers', 'c_person1technologyadoption', 'c_oldervsnewerbuilthomessco', 'c_healthfitnessmagbyrcat', 'c_estimatedcurrenthomevalue', 'c_buyeryoungadultclothshop', 'c_builthupcbuilt20002004', 'c_retailelectrongadg',
                          'c_actintcountrymusic', 'c_actintfitnessenthusiast', 'c_actinthealthyliving', 'c_actintoutdoorenthusiast', 'c_actintsportsenthusiast', 'c_actintpgatourenthusiast', 'c_educationindividual1', 'c_hhconsumerexpendituresshoes', 'c_newvsused', 'c_healthimage', 'c_workingcouples', 'c_hhconsumerexpendtravel', 'c_menscasualappareldollar', 'c_buyernondeptstoremakeup', 'c_ethnichhpchohwhitenhsp', 'c_buyerhighendspiritdrinker', 'c_actintsportsenthusiast', 'c_fincorporatecreditcardusr', 'c_educpop25pluspchsdiploma', 'c_buyerluxurystoreshop', 'c_esthhincomeamountv5', 'c_brandloyal', 'c_retailtravel']]

color_successes = 0
item_successes = 0

total = 0

print(len(test_data))
for index in range(len(test_data)):
    sample_data = testing_data.iloc[index, :]
    if (index == 50):
        print(sample_data)
    colors = np.dot(final_weights[0], np.transpose(sample_data))
    items = np.dot(final_weights[1], np.transpose(sample_data))

    colors = 100 / (1 + exp(-colors))
    items = 100 / (1 + exp(-items))
    colors = np.matrix.round(colors, 1)
    items = np.matrix.round(items, 1)

    predicted_Color = color_target_col[np.where(
        colors == np.amax(colors))[0][0]]
    new_a = np.delete(colors, np.where(colors == np.amax(colors))[0][0])
    nextpredicted_Color = color_target_col[np.where(
        new_a == np.amax(new_a))[0][0]]
    predicted_Item = item_target_col[np.where(items == np.amax(items))[0][0]]
    new_b = np.delete(items, np.where(items == np.amax(items))[0][0])
    nextpredicted_Item = item_target_col[np.where(
        new_b == np.amax(new_b))[0][0]]

    actual_color = test_data['color']
    actual_item = test_data['c_divisionname']

    if (predicted_Color == convertColor(actual_color[index+splitLine]) or nextpredicted_Color == convertColor(actual_color[index+splitLine]) or predicted_Item == convertItem(actual_item[index+splitLine]) or nextpredicted_Item == convertItem(actual_item[index+splitLine])):
        total += 1
        print("predictedColor: " + str(predicted_Color) + " | " + " actualColor: " + str(convertColor(actual_color[index+splitLine])) + " | predictedItem: " + str(
            predicted_Item) + " | " + " actualItem: " + str(convertItem(actual_item[index+splitLine])))

    if (predicted_Color == convertColor(actual_color[index+splitLine]) or nextpredicted_Color == convertColor(actual_color[index+splitLine])):
        color_successes += 1
        # print ("color success")
    if(predicted_Item == convertItem(actual_item[index+splitLine]) or nextpredicted_Item == convertItem(actual_item[index+splitLine])):
        item_successes += 1
        # print ("item success")

#print (color_successes/total)
#print (item_successes/total)


print("COLOR PREDICTION SUCCESS: " +
      str(color_successes/total))
print("ITEM PREDICTION SUCCESS: " + str(item_successes/total))
