import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


weekly_data = nfl.import_weekly_data([2020, 2021], ['player_name', 'recent_team', 'season', 'week',
                                                    'season_type', 'completions', 'attempts', 'passing_yards',
                                                    'passing_tds', 'interceptions', 'sacks',
                                                    'carries', 'rushing_yards', 'rushing_tds',
                                                    'receptions', 'targets', 'receiving_yards',
                                                    'receiving_tds', 'fantasy_points'], downcast=True)

data = pd.DataFrame(weekly_data)

categories = ['recent_team', 'season', 'week', 'season_type',
              'completions', 'attempts', 'passing_yards', 'passing_tds',
              'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds',
              'receptions', 'targets', 'receiving_yards', 'receiving_tds',
              'fantasy_points']

data_radar = data[['player_name'] + categories]

data_radar.to_csv('./data/weekly_data/2020-2021_games_weekly.csv')
