"""
Functions used to scrap teamrankings.com
"""

import numpy as np 
import pandas as pd
import difflib as dif
import matplotlib.pyplot as plt
import seaborn as sns

def get_rpi(date):
	"""
	Get rpi data from teamrankings.com and return it as a pandas
	dataframe.

	Parameters
	----------
	date : string ('yyyy-mm-dd')

	Returns
	-------
	df : pandas dataframe
		columns : Rank, Team, Rating, High, Low, Last
	"""

	link = 'https://www.teamrankings.com/ncb/rpi/?date=' + date
	rpi = pd.read_html(link)[0]

	# team record is in same column as team. remove it
	names = [t.split()[:-1] for t in rpi['Team']]
	names = [' '.join(n) for n in names]

	rpi['Team'] = names

	#compare the names to the kaggle competition given names
	return rpi


def name_cleaner(date):

	link = 'https://www.teamrankings.com/ncb/rpi/?date=' + date
	rpi = pd.read_html(link)[0]

	# team record is in same column as team. remove it
	names = [t.split()[:-1] for t in rpi['Team']]
	rpi_names = [' '.join(n) for n in names]
	rpi_names = list(rpi_names)
	rpi_names.sort()

	kag_names = pd.read_csv('../data/Teams.csv')['Team_Name']
	kag_names = list(kag_names)
	kag_names.sort()

	#find the closest matches to the kaggle names
	closest = [dif.get_close_matches(k,rpi_names,
		n=1,cutoff=0.0)[0] for k in kag_names]

	#print them out one by one and let the user decide
	verified = []
	for ii in range(len(kag_names)):

		s = kag_names[ii] + ' | ' + closest[ii]  

		inp = raw_input(s)

		if inp is 'y':
			verified.append(closest[ii])
		else:
			verified.append(np.nan)

		print ''

	return verified,kag_names

def draw_histogram():
	"""
	draw a histogram of the kaggle scores
	"""
	sns.set_context('notebook',font_scale=1.5)
	link = 'https://www.kaggle.com/c/march-machine-learning-mania-2016/leaderboard'
	tables = pd.read_html(link)
	df = tables[0].loc[1:].copy()
	df.drop([1,4,5],axis=1,inplace=True)
	df.columns=['Rank','Name','Score']

	#split up the names since they are all messy
	G = [df['Name'].iloc[ii].split()[0] for ii in range (len(df['Name'])) ]
	df['Name']=G

	dive_score = df[df['Name']=='UNCW'].Score.values.astype('float')
	dive_rank = df[df['Name']=='UNCW'].Rank.values.astype('int')

	scores = df['Score'].values.astype('float')

	#draw the plot
	fig,ax = plt.subplots(figsize=(10,5))
	f = ax.hist(scores[scores<1],25)

	pt1 = [dive_score,dive_score]
	pt2 = [0,np.max(f[0])]

	#put some text on it
	ax.plot(pt1,pt2)
	fig.text(.6,.8,'Dive Score = ' + str(dive_score[0]))
	fig.text(.6,.7,'Dive Rank = ' + str(dive_rank[0]))
	ax.set_xlabel('Log Loss')
	ax.set_ylabel('Count')

	return fig








