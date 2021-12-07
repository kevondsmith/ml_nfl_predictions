import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# nfl.import_seasonal_data([2007, 2008])
schedules = nfl.import_schedules([2020, 2021])

data = pd.DataFrame(schedules)
data.head()
data.columns

categories = ['season', 'game_type', 'week', 'gameday', 'weekday',
              'gametime', 'away_team', 'away_score', 'home_team', 'home_score',
              'result', 'total', 'total_line', 'temp', 'stadium']

data_radar = data[['game_id'] + categories]
data_radar.head()

data_radar.to_csv('./data/schedules/2020-2021_games_schedules.csv')
