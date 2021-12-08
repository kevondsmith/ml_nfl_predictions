import numpy as np,os,matplotlib.pyplot as plt,warnings
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model, neural_network
from sklearn import model_selection, metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


# Get data
dataset = pd.read_csv("~/CSCE 822 - Data Mining and Warehousing/Project/ml_nfl_predictions/data/schedules/2007-2008_games_schedules.csv")
# print(dataset.shape)

# May need to impute data
dataset = dataset[(dataset[dataset.columns] != "--").all(axis=1)]

# Remove all string features
dataset = dataset.loc[:, dataset.dtypes != str]
# print(dataset.shape)

# Try forcing to numeric
dataset = dataset.apply(lambda row: pd.to_numeric(row, errors='coerce', downcast='float'))
g = dataset.columns.to_series().groupby(dataset.dtypes).groups
# print(g)

# Remove all columns with NAs
dataset = dataset.dropna(axis = 1)

# Remove all columns that are not numeric
dataset = dataset._get_numeric_data()
print("Final shape = " + str(dataset.shape))
print("")

# Label cols
label = "result"
label_cols = ['away_team', 'away_score', 'home_team', 'home_score', 'result']


# Train/test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.20, random_state=42) # 20% test
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 20% val, 60% train

# Sweep found best max depth = 10
depths = [5, 10, 15]
mses = []
r2s = []
for depth in depths:
	# Try random forests
	regr = RandomForestRegressor(n_estimators=50, max_depth=depth,
	                                random_state=2)
	regr.fit(X_train, y_train)
	# Make predictions using the testing set
	y_pred = regr.predict(X_val)
	# The mean squared error
	print("Random Forest Mean squared error: %.2f"
	      % metrics.mean_squared_error(y_val, y_pred))
	mses.append(metrics.mean_squared_error(y_val, y_pred))
	# Explained variance score: 1 is perfect prediction
	print('Random Forest Variance score: %.2f' % metrics.r2_score(y_val, y_pred))
	print("")
	r2s.append(metrics.r2_score(y_val, y_pred))
	print("Best test score for k=", depth, "is: ", r2s, '\n'*2)

print("")
print("Random Forest depth sweep results:")

print('Depths:')
print(depths)
print('MSES:')
print(mses)
print('R2S:')
print(r2s)


# Sweep over num trees
ntrees = [10, 20, 50, 100]
mses = []
r2s = []
for ntree in ntrees:
	# Try random forests
	regr = RandomForestRegressor(n_estimators=ntree, max_depth=5,
	                                random_state=2)
	regr.fit(X_train, y_train)
	# Make predictions using the testing set
	y_pred = regr.predict(X_val)
	# The mean squared error
	print("")
	print("Random Forest Mean squared error: %.2f"
	      % metrics.mean_squared_error(y_val, y_pred))
	mses.append(metrics.mean_squared_error(y_val, y_pred))
	# Explained variance score: 1 is perfect prediction
	print('Random Forest Variance score: %.2f' % metrics.r2_score(y_val, y_pred))
	print("")
	r2s.append(metrics.r2_score(y_val, y_pred))

print("Random Forest ntrees sweep results:")
print('ntrees:')
print(ntrees)
print('MSES:')
print(mses)
print('R2S:')
print(r2s)





