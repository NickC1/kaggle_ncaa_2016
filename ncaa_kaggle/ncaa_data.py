"""
Puts the data into a format we will use to train models
"""
import numpy as np
import pandas as pd

class data_clean:
	"""
	Takes kaggle data in the form of CSV's and returns various stats
	and manipulated data.
	"""

	def __init__(self,year):

		self.year=year
		#read the data in
		f = '../data/RegularSeasonDetailedResults.csv'
		df = pd.read_csv(f)
		df = df[df.Season==year]

		#get the teams that played that year. Cant assume a team had a win
		self.teams = np.unique(df['Wteam'].append(df['Lteam']))

		#dont need season anymore and only need the daynumber
		self.game_stats = df.set_index('Daynum').drop('Season',axis=1)

		#get the unique days for storing the data
		self.unique_days = np.unique(df.Daynum)

		#initialize stat dictionary
		self.daily_stat_dict = dict()

		#post season stats
		tourney_df = pd.read_csv('../data/TourneyDetailedResults.csv')
		tourney_df = tourney_df[tourney_df.Season==year]
		self.tourney_data = tourney_df

	def paper_statistics(self):
		"""
		Insert the statistics as outlined in:
			'Predicting NCAAB match outcomes using MLtechniques: 
			some results and lessons learned'

		Inserts:
		Woe, Loe : offensive efficiency
		Wde, Lde : defensive efficiency
		Wefg, Lefg : effective field goal percentage
		Weto, Leto : effective turnover percentage
		Weor, Leor : effective offensive rebound percentage
		Weft, Left : effective free throw rate

		Parameters
		----------
		season : string; can either be 'regular' or 'post'

		Returns
		-------

		"""
		df = self.game_stats

		# Possessions
		df['Wposs'] = 0.96 * (df['Wfga'] - \
			df['Wor'] - df['Wto'] + .475*df['Wfta'])

		df['Lposs'] = 0.96 * (df['Lfga'] - \
			df['Lor'] - df['Lto'] + .475*df['Lfta']) 

		#offensive efficiency
		df['Woe'] = df['Wscore']/df['Wposs']
		df['Loe'] = df['Lscore']/df['Lposs']

		#defensive efficiency
		df['Wde'] = df['Lscore']/df['Lposs']
		df['Lde'] = df['Wscore']/df['Wposs']

		#effective field goal
		topw = df['Wfgm'] + .5* df['Wfgm3']
		bottomw = df['Wfga'] + df['Wfga3']
		df['Wefg'] = topw/bottomw

		topl = df['Lfgm'] + .5* df['Lfgm3']
		bottoml = df['Lfga'] + df['Lfga3']
		df['Lefg'] = topl/bottoml

		#effective turnovers
		df['Weto'] = df['Wto']/df['Wposs']
		df['Leto'] = df['Lto']/df['Lposs']

		#effective offensive rebounds
		df['Weor'] = df['Wor']/(df['Wor']+df['Ldr'])
		df['Leor'] = df['Lor']/(df['Lor']+df['Wdr'])

		#effective free throw rate
		df['Weftr'] = df['Wfta']/df['Wfga']
		df['Leftr'] = df['Lfta']/df['Lfga']

		self.game_stats = df

		print('Paper Stats Added!')


	def daily_schedule(self):
		"""
		Returns the daily schedule for each team.


		Returns
		-------
		df : DataFrame
			index is the day of the season and column is the
			the team. The values are the matrix is who the team
			played on each specific day.
		"""

		X = self.game_stats[['Wteam','Lteam']]
		
		#store the wins and losses. 
		store_sched = pd.DataFrame(index = self.unique_days,
			columns=self.teams).fillna(value=0)

		for ii in range(len(X.index)):

			day = X.index[ii]
			row = X.iloc[ii]

			w_team = row['Wteam']#.values #winning team
			l_team = row['Lteam']#.values #losing team

			#looking for who they played
			store_sched.loc[day,w_team] = l_team
			store_sched.loc[day,l_team] = w_team

		self.schedule = store_sched

		print('Schedule has been calculated!')


	def stat_select(self,stat):
		"""
		Gets the stats for all the teams for each day in the season.

		Parameters
		----------
		year : int , which year to grab data
		stat : string
		Can be:
		['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr', 'ast',
		 'to', 'stl', 'blk', 'pf', 'poss', 'oe', 'de', 'efg', 'eto', 'eor', 'eftr']
		 Not implemented:
		 'rpi','rpi_1', rpi_2, 'rpi_3'

		Returns
		-------
		store_feature : dataframe with columns the team and rows the days
			values are the stats scored in the game
		store_outcome : dataframe with columns the team and rows the days
			values are 1 for win or 0 for loss.
		"""

		X = self.game_stats
	
		store_feature = pd.DataFrame(index = self.unique_days,
			columns=self.teams)

		#how it is represented in the game_data
		w_f = 'W' + stat
		l_f = 'L' + stat

		#loop through line and put the stat in correct places
		for ii in range(len(X.index)):

			day = X.index[ii]
			row = X.iloc[ii]

			w_team = row['Wteam'] #winning team
			w_feature = row[w_f]
			l_team = row['Lteam'] #losing team
			l_feature = row[l_f]

			store_feature.loc[day,w_team] = w_feature
			store_feature.loc[day,l_team] = l_feature

		# take the average of it through the season
		store_feature = pd.expanding_mean(store_feature)

		self.daily_stat_dict[stat] = store_feature
		
		print(stat + ' saved to daily_stat_dict!')


	def daily_win_percentage(self):
		"""
		Gets the winning percentage of each team through the season.

		Returns
		-------
		win_percent : winning percentage through time
		"""

		X = self.game_stats
	
		store_outcome = pd.DataFrame(index = self.unique_days,
			columns=self.teams)

		for ii in range(len(X.index)):

			row = X.iloc[ii]
			day = X.index[ii]

			w_team = row['Wteam'] #winning team
			l_team = row['Lteam'] #losing team

			store_outcome.loc[day,w_team] = 1
			store_outcome.loc[day,l_team] = 0


		win_percent = pd.expanding_mean(store_outcome)

		self.daily_stat_dict['win_perc'] = win_percent
		print('win_perc saved to daily_stat_dict!')

	def daily_win_percentage2(self):

		"""
		Calculates the second column of the rpi scores
		Inputs the winning percentage of the team played
		on that day
		"""

		perc = self.daily_stat_dict['win_perc']
		sched = self.schedule

		store_perc = pd.DataFrame(index = self.unique_days,
			columns=self.teams)

		for ii in range(len(perc.index)):

			# fill in nans for the first day since no teams have a 
			# winning percentage yet
			if ii ==0:
				ind0 = perc.index[ii]
				store_perc.loc[ind0] = np.nan

			else:
				ind0 = perc.index[ii]
				ind1 = perc.index[ii-1]
				p = perc.loc[ind1]		# percentages of team before that point
				s = sched.loc[ind0].values 	# teams played


				team_played = s[s!=0] #teams played

				ind_team = perc.columns.values[s!=0].astype(int) # get the teams
				
				store_perc.loc[ind0][ind_team] = p[team_played].values

		#now go through and have a running average of the teams played
		store_perc = pd.expanding_mean(store_perc)

		self.daily_stat_dict['win_perc2'] = store_perc
		print('win_perc2 saved to daily_stat_dict!')

	def daily_win_percentage3(self):

		"""
		Calculates the third column of the rpi scores
		Inputs the winning percentage of the team played
		on that day
		"""

		perc = self.daily_stat_dict['win_perc2']
		sched = self.schedule

		store_perc = pd.DataFrame(index = self.unique_days,
			columns=self.teams)

		for ii in range(len(perc.index)):

			# fill in nans for the first day since no teams have a 
			# winning percentage yet
			if ii ==0:
				ind0 = perc.index[ii]
				store_perc.loc[ind0] = np.nan

			else:
				ind0 = perc.index[ii]
				ind1 = perc.index[ii-1]
				p = perc.loc[ind1]		# percentages of team before that point
				s = sched.loc[ind0].values 	# teams played


				team_played = s[s!=0] #teams played

				ind_team = perc.columns.values[s!=0].astype(int) # get the teams
				
				store_perc.loc[ind0][ind_team] = p[team_played].values

		#now go through and have a running average of the teams played
		store_perc = pd.expanding_mean(store_perc)

		self.daily_stat_dict['win_perc3'] = store_perc
		print('win_perc3 saved to daily_stat_dict!')

	def rpi(self):
		"""
		Calculates the rpi value based on:
		win_perc, win_perc2, win_perc3
		"""

		p1 = self.daily_stat_dict['win_perc']
		p2 = self.daily_stat_dict['win_perc2']
		p3 = self.daily_stat_dict['win_perc3']

		self.daily_stat_dict['rpi'] = .5*p1 + .25*p2 + .5*p3
		print('rpi saved to daily_stat_dict!')


	def end_of_season_stats(self):
		"""
		Retrieves the end of season statistics for each of the stats
		in self.daily_stat_dict. The data returned from this are used
		to predict the post season.
		"""

		stat_names = self.daily_stat_dict.keys()
		stats = self.daily_stat_dict

		eos_stats = pd.DataFrame(index = self.teams,
			columns=stat_names)

		for s in stat_names:

			daily = stats[s].values[-1,:] #last row

			eos_stats[s] = daily

		self.eos_stats = eos_stats
		print 'end of season stats calculated (eos_stats)'

	
	def tourney_features(self):
		"""
		Construct the tournament features from the end of season stats
		"""
		eos = self.eos_stats #ind is team_id, cols are stats
		td = self.tourney_data

		#create our dataframe
		cnames = eos.columns.values
		c1 = [n + '_1' for n in cnames]
		c2 = [n + '_2' for n in cnames]
		cnames = c1 + c2

		f1 = [] #store the features for a vstack at the end
		f2 = [] # for both permuations
		for ii in range(len(td.index)):

			row = td.iloc[ii]
			wteam = row['Wteam']
			lteam = row['Lteam']

			w_feats = eos.loc[wteam].values[np.newaxis]
			l_feats = eos.loc[lteam].values[np.newaxis]

			f1.append( np.hstack((w_feats,l_feats)) )
			f2.append( np.hstack((l_feats,w_feats)) )

		f1 = np.vstack(f1)
		f2 = np.vstack(f2)
		
		t1 = np.ones(len(f1)).reshape(len(f1),1)
		t2 = np.zeros(len(f2)).reshape(len(f2),1)

		features = np.vstack((f1,f2))
		targets = np.vstack((t1,t2))

		self.tourney_f_data = pd.DataFrame(features,columns=cnames)
		self.tourney_t_data = pd.DataFrame(targets)

		print('Features stored in tourney_f_data')
		print('Targets stored in tourney_t_data')

	def season_features(self):
		"""
		***THIS ISNT FINISHED****
		Gets the features for the games within the season.
		This is different than tourney features which fills in the
		end of season stats

		"""

		stat_names = self.daily_stat_dict.keys()
		stats = self.daily_stat_dict

		X = self.game_stats

		for s_n in stat_names:

			daily_stat = stats[s_n]

	def ids_to_names(self,df,id='ind'):
		"""
		Converts the columns or the indices to team names.
		This is useful to make sure our RPI is working correctly.

		Parameters
		----------
		df : dataframe where we want to convert the ind or columns
		id : where the id is located either 'ind' or 'col'
		"""

		teams = pd.read_csv('../data/Teams.csv')
		teams.set_index('Team_Id',inplace=True)
		team_dict = teams.to_dict()['Team_Name'] #remove the higher level

		X = df.copy()

		if id=='ind':
			ids = X.index.values
		if id=='cols':
			ids = X.columns.values

		names = [team_dict[t_id] for t_id in ids]

		X.index=names #put them with the best teams at top

		return X

	def submission_df(self):
		"""
		Gets a clean dataframe from the sample submission file
		"""

		samp = pd.read_csv('../data/SampleSubmission.csv')


		yr = []
		t1=[]
		t2=[]
		for ii in range(len(samp)):
			row = samp['Id'][ii].split('_')
			yr.append(row[0])
			t1.append(row[1])
			t2.append(row[2])

		yr = np.array(yr)[:,np.newaxis]
		t1 = np.array(t1)[:,np.newaxis]
		t2 = np.array(t2)[:,np.newaxis]

		s = np.hstack((yr,t1,t2)).astype(int)
		sub_df = pd.DataFrame(s,columns=['year','team1','team2'])
		sub_df = sub_df[sub_df['year']==self.year]

		self.sub_df = sub_df

		print("Submission dataframe created as sub_df")

	def submission_features(self):
		"""
		Construct the submission features from the end of season stats
		Note that this only works for years 2012-2015
		"""
		eos = self.eos_stats #ind is team_id, cols are stats
		td = self.sub_df

		#create our dataframe
		cnames = eos.columns.values
		c1 = [n + '_1' for n in cnames]
		c2 = [n + '_2' for n in cnames]
		cnames = c1 + c2

		f1 = [] #store the features for a vstack at the end
		for ii in range(len(td.index)):

			row = td.iloc[ii]
			team1 = row['team1']
			team2 = row['team2']

			team1_feats = eos.loc[team1].values[np.newaxis]
			team2_feats = eos.loc[team2].values[np.newaxis]

			f1.append( np.hstack((team1_feats,team2_feats)) )

		f1 = np.vstack(f1)

		self.submission_f_data = pd.DataFrame(f1,columns=cnames)

		print('Submission stored in submission_f_data')	








class KenPom:
	"""
	Works with the KenPom data. Currently there is only 2014 data
	"""

	def __init__(self):

		#ken pom end of season data
		kp_f = '../data/kp-2014.csv'
		kp_df = pd.read_csv(kp_f)

		#tournament data for that year
		year=2014
		t_f = '../data/TourneyDetailedResults.csv'
		tourney_df = pd.read_csv(t_f)
		self.tourney_df = tourney_df[tourney_df.Season==year]

		#ken pom to kaggle ids
		ids = pd.read_csv('../data/kp-2016-name-kaggleId.csv',index_col=False)
		
		# cols: Team, Tempo, PosLO, PosLD, OE, DE, Team_Name, Rank, Team_Id
		data = pd.merge(kp_df,ids,left_on='Team',right_on='Team_Name')
		data.set_index('Team_Id',inplace=True)
		self.data = data
	def tourney_features(self):
		"""
		Construct the tournament features from the end of season stats
		"""
		stats = self.data[['Tempo', 'PosLO', 'PosLD', 'OE', 'DE']] #ind is team_id, cols are stats
		td = self.tourney_df

		#create our dataframe
		cnames = stats.columns.values
		c1 = [n + '_1' for n in cnames]
		c2 = [n + '_2' for n in cnames]
		cnames = c1 + c2

		f1 = [] #store the features for a vstack at the end
		f2 = [] # for both permuations
		for ii in range(len(td.index)):

			row = td.iloc[ii]
			wteam = row['Wteam']
			lteam = row['Lteam']

			w_feats = stats.loc[wteam].values[np.newaxis]
			l_feats = stats.loc[lteam].values[np.newaxis]

			f1.append( np.hstack((w_feats,l_feats)) )
			f2.append( np.hstack((l_feats,w_feats)) )

		f1 = np.vstack(f1)
		f2 = np.vstack(f2)
		
		t1 = np.ones(len(f1)).reshape(len(f1),1)
		t2 = np.zeros(len(f2)).reshape(len(f2),1)

		features = np.vstack((f1,f2))
		targets = np.vstack((t1,t2))

		self.tourney_f_data = pd.DataFrame(features,columns=cnames)
		self.tourney_t_data = pd.DataFrame(targets)

		print('Features stored in tourney_f_data')
		print('Targets stored in tourney_t_data')








