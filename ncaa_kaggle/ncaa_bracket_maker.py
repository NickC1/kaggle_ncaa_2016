
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class bracket_create:
	"""
	Takes the probabilities from the csv and creates a 
	bracket with it filled in with the probabilities.
	"""

	def __init__(self, path_to_csv):
		"""
		touney_teams was preprocessed so that the team names match
		what kaggle provides.
		"""

		self.prob_df = pd.read_csv(path_to_csv)
		self.team_ids = dict()
		self.touney_teams = dict()

		#teams on the left or right
		self.left_teams = dict()
		self.right_teams = dict()
		self.L_ids = dict()
		self.R_ids = dict()

		#probabilities for the teams
		self.left_probs = dict()
		self.right_probs = dict()

		#set up the plotting
		fig,ax = plt.subplots(figsize=(20,10))
		ax.set_axis_off()
		self.fig=fig

		self.tourney_teams = dict()
		self.tourney_teams['round1'] = ['Austin Peay','Kansas',
			'Colorado', 'Connecticut',
			'Maryland', 'S Dakota St',
			'California', 'Hawaii',
			'Arizona', 'Vanderbilt',
			'Buffalo', 'Miami FL',
			'Iowa', 'Temple',
			'UNC Asheville', 'Villanova',
			'Holy Cross', 'Oregon',
			'Cincinnati', "St Joseph's PA",
			'Baylor', 'Yale',
			'Duke', 'UNC Wilmington',
			'Northern Iowa', 'Texas',
			'Texas A&M', 'WI Green Bay',
			'Oregon St', 'VA Commonwealth',
			'CS Bakersfield', 'Oklahoma',
			'F Dickinson', 'North Carolina',
			'Providence', 'USC',
			'Chattanooga', 'Indiana',
			'Kentucky', 'Stony Brook',
			'Michigan', 'Notre Dame',
			'SF Austin', 'West Virginia',
			'Pittsburgh', 'Wisconsin',
			'Weber St', 'Xavier',
			'Hampton', 'Virginia',
			'Butler', 'Texas Tech',
			'Ark Little Rock', 'Purdue',
			'Iona', 'Iowa St',
			'Gonzaga', 'Seton Hall',
			'Fresno St', 'Utah',
			'Dayton', 'Syracuse',
			'Michigan St', 'MTSU']

		self.tourney_teams['round2'] = [ 'Connecticut', 'Kansas',
			'California', 'Maryland',
			'Arizona', 'Miami FL',
			'Iowa', 'Villanova',
			'Oregon', "St Joseph's PA",
			'Baylor', 'UNC Wilmington',
			'Texas','Texas A&M', 
			'Oklahoma', 'VA Commonwealth', #split
			'North Carolina', 'Providence',
			'Indiana', 'Kentucky',
			'Notre Dame', 'West Virginia',
			'Pittsburgh', 'Xavier',
			'Butler', 'Virginia',
			'Iowa St', 'Purdue',
			'Seton Hall', 'Utah',
			'Dayton', 'Michigan St']

		self.tourney_teams['round3'] = ['California','Kansas',
		'Miami FL', 'Villanova',
		'Baylor','Oregon',
		'Oklahoma','Texas A&M', #split
		'Indiana','North Carolina',
		'West Virginia', 'Xavier',
		'Purdue', 'Virginia',
		'Michigan St','Utah']

		self.tourney_teams['round4'] = ['Kansas','Villanova',
		'Oklahoma', 'Oregon',
		'North Carolina','Xavier',
		'Michigan St', 'Virginia']


	def draw_everything(self):
		"""
		Calls all of the other functions. Its an all in one!
		"""

		for rnd in ['round1','round2','round3','round4']:
		
			self.team_ids_from_name(rnd)
			self.matches(rnd)
			self.probabilities(rnd)

		self.plot_round1()
		self.plot_round2()
		self.plot_round3()
		self.plot_round4()

		print("call .fig to get the bracket!")


	def team_ids_from_name(self,round_num):
		"""
		get the team ids given a name.
		round_num : 'round1' or 'round2'
		"""

		kg = pd.read_csv('../data/Teams.csv')
		# get their corresponding ids
		ids = []
		tourney_teams = self.tourney_teams[round_num]

		for tn in tourney_teams:
			id_mask =  kg['Team_Name']==tn
			ids.append(kg['Team_Id'][id_mask].values[0])

		self.team_ids[round_num] = ids
		print('IDs matched')

	def matches(self,round_num):
		"""
		Split the team up by their matches
		"""

		tourney_teams = self.tourney_teams[round_num]
		ids = self.team_ids[round_num]

		left_teams = []
		right_teams = []
		L_ids = []
		R_ids = []

		len_teams = len(tourney_teams)/2
		for ii in range(0,len_teams,2):
		
			left_teams.append( tourney_teams[ii:ii+2] )
			right_teams.append( tourney_teams[ii+len_teams:ii+len_teams+2] )

			L_ids.append( ids[ii:ii+2] )
			R_ids.append( ids[ii+len_teams:ii+len_teams+2] )

		self.left_teams[round_num] = left_teams
		self.right_teams[round_num] = right_teams
		self.L_ids[round_num] = L_ids
		self.R_ids[round_num] = R_ids

		print('Left and Right organized!')

	def probabilities(self,round_num):

		L_ids = self.L_ids[round_num]
		R_ids = self.R_ids[round_num]
		#strings for table lookup
		L_lookup = ['2016_' + str(Li[0]) + '_' + str(Li[1]) for Li in L_ids]
		R_lookup = ['2016_' + str(Ri[0]) + '_' + str(Ri[1]) for Ri in R_ids]

		#probs
		prob_df = self.prob_df

		L_p = []
		R_p = []
		for ii in range(len(L_lookup)):
			L_mask = prob_df.Id == L_lookup[ii]
			R_mask = prob_df.Id == R_lookup[ii]

			L_p.append( prob_df.Pred[L_mask].values[0] )
			R_p.append( prob_df.Pred[R_mask].values[0] )

		self.left_probs[round_num] = L_p
		self.right_probs[round_num] = R_p
		print("probabilities calculated!")

	def plot_round1(self):
		
		
		left_teams = self.left_teams['round1']
		right_teams = self.right_teams['round1']
		l_probs = self.left_probs['round1']
		r_probs = self.right_probs['round1']


		for ii in range(len(left_teams)):
			lt = left_teams[15-ii]
			rt = right_teams[15-ii]
			#Round 1
			#left side
			lp = round(l_probs[15-ii],2)
			rp = round(r_probs[15-ii],2)

			self.fig.text(.1,(1/16.)*ii,lt[1] + ' ' + str(1 - lp) ) 
			self.fig.text(.1,(1/16.)*ii +.02,lt[0] + ' ' + str(lp))
			#right side
			self.fig.text(.9, (1/16.)*ii, rt[1]+ ' ' + str(1-rp) )
			self.fig.text(.9,(1/16.)*ii +.02,rt[0] + ' ' + str(rp) )

		print('round 1 printed')

	def plot_round2(self):
		
		
		left_teams = self.left_teams['round2']
		right_teams = self.right_teams['round2']
		l_probs = self.left_probs['round2']
		r_probs = self.right_probs['round2']


		for ii in range(len(left_teams)):
			lt = left_teams[7-ii]
			rt = right_teams[7-ii]
			#Round 1
			#left side
			lp = round(l_probs[7-ii],2)
			rp = round(r_probs[7-ii],2)

			self.fig.text(.2,(1/8.)*ii + .025,lt[1] + ' ' + str(1 - lp) ) 
			self.fig.text(.2,(1/8.)*ii +.055,lt[0] + ' ' + str(lp))
			#right side
			self.fig.text(1-.2, (1/8.)*ii+.025, rt[1]+ ' ' + str(1-rp) )
			self.fig.text(1-.2,(1/8.)*ii +.055,rt[0] + ' ' + str(rp) )

		print('round 2 printed!')

	def plot_round3(self):

		left_teams = self.left_teams['round3']
		right_teams = self.right_teams['round3']
		l_probs = self.left_probs['round3']
		r_probs = self.right_probs['round3']


		for ii in range(len(left_teams)):
			lt = left_teams[3-ii]
			rt = right_teams[3-ii]
			#Round 1
			#left side
			lp = round(l_probs[3-ii],2)
			rp = round(r_probs[3-ii],2)

			self.fig.text(.3,(1/4.)*ii +.1 ,lt[1] + ' ' + str(1 - lp) ) 
			self.fig.text(.3,(1/4.)*ii +.13,lt[0] + ' ' + str(lp))
			#right side
			self.fig.text(1-.3, (1/4.)*ii + .1, rt[1]+ ' ' + str(1-rp) )
			self.fig.text(1-.3,(1/4.)*ii + .13,rt[0] + ' ' + str(rp) )

		print('round 3 printed!')

	def plot_round4(self):

		left_teams = self.left_teams['round4']
		right_teams = self.right_teams['round4']
		l_probs = self.left_probs['round4']
		r_probs = self.right_probs['round4']


		for ii in range(len(left_teams)):
			lt = left_teams[1-ii]
			rt = right_teams[1-ii]
			#Round 1
			#left side
			lp = round(l_probs[1-ii],2)
			rp = round(r_probs[1-ii],2)

			self.fig.text(.4,(1/2.)*ii+ .25,lt[1] + ' ' + str(1 - lp) ) 
			self.fig.text(.4,(1/2.)*ii +.28,lt[0] + ' ' + str(lp))
			#right side
			self.fig.text(1-.4, (1/2.)*ii+ .25, rt[1]+ ' ' + str(1-rp) )
			self.fig.text(1-.4,(1/2.)*ii +.28,rt[0] + ' ' + str(rp) )

		print('round 4 printed!')

		

    
    






