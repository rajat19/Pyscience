import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import seaborn as sns
from plotly.graph_objs import *

Global = pd.read_csv('datasets/religions/global.csv')
national = pd.read_csv('datasets/religions/national.csv')
regional = pd.read_csv('datasets/religions/regional.csv')

national_2010 = national[national['year'] == 2010]

national_2010.rename(columns={
	'baha\xe2\x80\x99i_percent': 'bahai_percent',
	'shi\xe2\x80\x99a_percent': 'shia_percent'
}, inplace=True)
religion_list = []
for col in national_2010.columns:
	if '_percent' in col:
		religion_list.append(col)

# print(religion_list)
# considering only major world religions

national_2010['major_religion'] = national_2010[['christianity_percent', 'judaism_percent', 'islam_percent', 'buddhism_percent', 'zoroastrianism_percent', 'hinduism_percent', 'sikhism_percent', 'shinto_percent', 'bahai_percent', 'taoism_percent', 'jainism_percent', 'confucianism_percent']].idxmax(axis=1)
# national_2010['major_religion'] = national_2010[['protestant_percent', 'romancatholic_percent', 'easternorthodox_percent', 'anglican_percent', 'otherchristianity_percent', 'orthodox_percent', 'conservative_percent', 'reform_percent', 'otherjudaism_percent', 'sunni_percent', 'shia_percent', 'ibadhi_percent', 'nationofislam_percent', 'alawite_percent', 'ahmadiyya_percent', 'otherislam_percent', 'mahayana_percent', 'theravada_percent', 'otherbuddhism_percent', 'zoroastrianism_percent', 'hinduism_percent', 'sikhism_percent', 'shinto_percent', 'bahai_percent', 'taoism_percent', 'jainism_percent', 'confucianism_percent', 'syncretism_percent', 'animism_percent', 'noreligion_percent', 'otherreligion_percent', 'total_percent']]

# map values to proper english keywords
religion_mapping = {
	'christianity_percent': 'Christianity',
	'judaism_percent': 'Judaism',
	'islam_percent': 'Islam',
	'buddhism_percent': 'Buddhism',
	'zoroastrianism_percent': 'Zoroastrianism',
	'hinduism_percent': 'Hinduism',
	'sikhism_percent': 'Sikhism',
	'shinto_percent': 'Shintoism',
	'bahai_percent': 'Bahai',
	'taoism_percent': 'Taoism',
	'jainism_percent': 'Jainism',
	'confucianism_percent': 'Confucianism',
	'syncretism_percent': 'Syncretism',
	'animism_percent': 'Animism',
	'noreligion_percent': 'No religion',
	'otherreligion_percent': 'Other religion'
}
national_2010['major_religion'] = national_2010['major_religion'].map(religion_mapping)

# print(national_2010['major_religion'].unique())
# print(national_2010[['state', 'code', 'population', 'major_religion']].sort_values(by='population', ascending=False).head(10))

trace_christianity = Choropleth(
	z=national_2010[national_2010['major_religion'] == 'Christianity']['christianity_percent'].values,
	autocolorscale=False,
	colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(0, 0, 120)']],
	hoverinfo='text',
	locationmode = 'country names',
	locations=national_2010[national_2010['major_religion'] == 'Christianity']['state'].values,
	name='Christianity',
	showscale=False,
	text=national_2010[national_2010['major_religion'] == 'Christianity'][['state', 'christianity_percent']].values,
	zauto=False,
	zmax=1,
    zmin=0,
	marker = dict(
		line = dict(
			color = 'rgb(200,200,200)',
			width = 0.5
		)
	)
)

trace_islam = Choropleth(
	z=national_2010[national_2010['major_religion'] == 'Islam']['islam_percent'].values,
	autocolorscale=False,
	colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(51, 160, 44)']],
	hoverinfo='text',
	locationmode = 'country names',
	locations=national_2010[national_2010['major_religion'] == 'Islam']['state'].values,
	name='Islam',
	showscale=False,
	text=national_2010[national_2010['major_religion'] == 'Islam'][['state', 'islam_percent']].values,
	zauto=False,
	zmax=1,
    zmin=0,
	marker = dict(
		line = dict(
			color = 'rgb(200,200,200)',
			width = 0.5
		)
	)
)

trace_hinduism = Choropleth(
	z=national_2010[national_2010['major_religion'] == 'Hinduism']['hinduism_percent'].values,
	autocolorscale=False,
	colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(230, 115, 0)']],
	hoverinfo='text',
	locationmode = 'country names',
	locations=national_2010[national_2010['major_religion'] == 'Hinduism']['state'].values,
	name='Hinduism',
	showscale=False,
	text=national_2010[national_2010['major_religion'] == 'Hinduism'][['state', 'hinduism_percent']].values,
	zauto=False,
	zmax=1,
    zmin=0,
	marker = dict(
		line = dict(
			color = 'rgb(200,200,200)',
			width = 0.5
		)
	)
)

trace_judaism = Choropleth(
	z=national_2010[national_2010['major_religion'] == 'Judaism']['judaism_percent'].values,
	autocolorscale=False,
	colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(119, 51, 255)']],
	hoverinfo='text',
	locationmode = 'country names',
	locations=national_2010[national_2010['major_religion'] == 'Judaism']['state'].values,
	name='Judaism',
	showscale=False,
	text=national_2010[national_2010['major_religion'] == 'Judaism'][['state', 'judaism_percent']].values,
	zauto=False,
	zmax=1,
    zmin=0,
	marker = dict(
		line = dict(
			color = 'rgb(200,200,200)',
			width = 0.5
		)
	)
)

trace_buddhism = Choropleth(
	z=national_2010[national_2010['major_religion'] == 'Buddhism']['buddhism_percent'].values,
	autocolorscale=False,
	colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(179, 179, 0)']],
	hoverinfo='text',
	locationmode = 'country names',
	locations=national_2010[national_2010['major_religion'] == 'Buddhism']['state'].values,
	name='Buddhism',
	showscale=False,
	text=national_2010[national_2010['major_religion'] == 'Buddhism'][['state', 'buddhism_percent']].values,
	zauto=False,
	zmax=1,
    zmin=0,
	marker = dict(
		line = dict(
			color = 'rgb(200,200,200)',
			width = 0.5
		)
	)
)

trace_confucianism = Choropleth(
	z=national_2010[national_2010['major_religion'] == 'Confucianism']['confucianism_percent'].values,
	autocolorscale=False,
	colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(255, 0, 0)']],
	hoverinfo='text',
	locationmode = 'country names',
	locations=national_2010[national_2010['major_religion'] == 'Confucianism']['state'].values,
	name='Confucianism',
	showscale=False,
	text=national_2010[national_2010['major_religion'] == 'Confucianism'][['state', 'confucianism_percent']].values,
	zauto=False,
	zmax=1,
    zmin=0,
	marker = dict(
		line = dict(
			color = 'rgb(200,200,200)',
			width = 0.5
		)
	)
)

trace_shintoism = Choropleth(
	z=national_2010[national_2010['major_religion'] == 'Shintoism']['shinto_percent'].values,
	autocolorscale=False,
	colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(204, 0, 102)']],
	hoverinfo='text',
	locationmode = 'country names',
	locations=national_2010[national_2010['major_religion'] == 'Shintoism']['state'].values,
	name='Shintoism',
	showscale=False,
	text=national_2010[national_2010['major_religion'] == 'Shintoism'][['state', 'shinto_percent']].values,
	zauto=False,
	zmax=1,
    zmin=0,
	marker = dict(
		line = dict(
			color = 'rgb(200,200,200)',
			width = 0.5
		)
	)
)

data = Data([trace_christianity, trace_islam, trace_hinduism, trace_buddhism, trace_confucianism, trace_judaism, trace_shintoism])
# data = [
# 	dict(
# 		type = 'choropleth',
# 		autocolorscale = False,
# 		colorscale = 'Viridis',
# 		reversescale = True,
# 		showscale = True,
# 		locations = national_2010['state'].values,
# 		z = national_2010['islam_percent'].values,
# 		locationmode = 'country names',
# 		text = national_2010['state'].values,
# 		marker = dict(
# 			line = dict(
# 				color = 'rgb(200,200,200)',
# 				width = 0.5
# 			)
# 		),
# 		colorbar = dict(
# 			autotick = True,
# 			tickprefix = '', 
#           title = 'Number of Islam Adherents'
# 		)
# 	)
# ]

layout = Layout(
    title = 'Major Religions',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,0,0)',
        projection = dict(
        	type = 'orthographic',
            rotation = dict(
				lon = 60,
				lat = 10
			),
        ),
        lonaxis =  dict(
			showgrid = False,
			gridcolor = 'rgb(102, 102, 102)'
		),
        lataxis = dict(
			showgrid = False,
			gridcolor = 'rgb(102, 102, 102)'
		),
    ),
	showlegend = True,
)
fig = dict(data=data, layout=layout)
py.plot(fig, validate=False, filename='images/worldreligionmap2010')

plt.show()