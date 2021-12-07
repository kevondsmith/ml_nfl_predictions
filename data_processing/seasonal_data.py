import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


seasonal_data = nfl.import_seasonal_data([2020, 2021])

data = pd.DataFrame(seasonal_data)

categories = ['season', 'season_type', 'completions',
              'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'passing_air_yards',
              'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'receptions',
              'targets', 'receiving_yards', 'receiving_tds',
              'receiving_yards_after_catch', 'fantasy_points']

df = data[['player_name'] + categories]

df.to_csv('./data/seasonal_data/2020-2021_games_seasonal.csv')
