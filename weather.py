import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from pylab import rcParams

data = pd.read_csv('datasets/weather/testset.csv', parse_dates=['datetime_utc'], skipinitialspace=True)

# print(data.columns)

# Formatted Date transformation:
data['Date'] = pd.to_datetime(data['datetime_utc'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['hour'] = data['Date'].dt.hour

yearwise_data = data.groupby(data.year).mean()

# Delhi average humidity by year
pd.stats.moments.ewma(yearwise_data._hum, 5).plot()
yearwise_data._hum.plot(linewidth=1)
plt.title('Delhi Average Humidity by year')
plt.xlabel('year')

# Delhi average heat by year
pd.stats.moments.ewma(yearwise_data._heatindexm, 5).plot()
yearwise_data._heatindexm.plot(linewidth=1)
plt.title('Delhi Average Heat by year')
plt.xlabel('year')

# Delhi average rain by year
pd.stats.moments.ewma(yearwise_data._rain, 5).plot()
yearwise_data._rain.plot(linewidth=1)
plt.title('Delhi Average Rain by year')
plt.xlabel('year')

# Delhi heat
p = sns.stripplot(data=data, x='year', y='_heatindexm')
p.set(title='Delhi heat')
dec_ticks = [y if not x%20 else '' for x, y in enumerate(p.get_xticklabels())]
p.set(xticklabels=dec_ticks)

# Drawing a heatmap
def facet_heatmap(data, color, **kws):
    values = data.columns.values[3]
    data = data.pivot(index='day', columns='hour', values=values)
    sns.heatmap(data, cmap='coolwarm', **kws)

def weather_calendar(year, weather):
    dfyear = data[data['year'] == year][['month', 'day', 'hour', weather]]
    vmin = dfyear[weather].min()
    vmax = dfyear[weather].max()
    with sns.plotting_context(font_scale=12):
        g = sns.FacetGrid(dfyear, col='month', col_wrap=3)
        g = g.map_dataframe(facet_heatmap, vmin=vmin, vmax=vmax)
        g.set_axis_labels('Hour', 'Day')
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle('%s Calendar, Year: %s.' %(weather, year), fontsize=18)

weather_calendar(2006, '_hum')
weather_calendar(2006, '_rain')
weather_calendar(2006, '_fog')
plt.show()