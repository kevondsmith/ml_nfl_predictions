import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


win_totals = nfl.import_win_totals([2020, 2021])
data = pd.DataFrame(win_totals)
data.sort_values(by=['season'])

data.to_csv('./data/win_totals/2020-2021_win_totals.csv')
