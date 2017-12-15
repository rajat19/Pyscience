import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def label_winner(row):
    if row['hs'] > row['as']:
        return row['ht']
    if row['hs'] < row['as']:
        return row['at']
    return 'Draw'

def label_era(x):
    year = int(x[:str(x).find('-')])
    if year > 2000:
        return 'Modern(>2000)'
    if year > 1976:
        return 'Middling(>1976)'
    return 'Old(<1976)'


results = pd.read_csv("datasets/football_events/results.csv")

# print(results.isnull().any().any())
# rename columns to be more readable
results.columns = ['date', 'ht', 'at', 'hs', 'as', 'tnmt', 'city', 'country']
# print(results.head())
# print(results.tnmt.unique())
# print(results.groupby(results.ht).mean().sort_values(by="hs", ascending=False))

results['winner'] = results.apply(lambda row: label_winner (row),axis=1)
results['era'] = results['date'].apply(lambda x: label_era(x))

# Count of different tournaments
tournamentCount = results.groupby(by=['tnmt'])['at'].agg({'Count': np.size})
tournamentCount['Count'] = tournamentCount['Count'].astype(int)
tournamentCount = tournamentCount.sort_values(by="Count", ascending=False)
# print(tournamentCount.head())

homeTeamCount = results.groupby(by=['ht'])['tnmt'].agg({'Count': np.size})
homeTeamCount['Count'] = homeTeamCount['Count'].astype(int)
homeTeamCount = homeTeamCount.sort_values(by = 'Count', ascending=False)
# print(homeTeamCount.head())

awayTeamCount = results.groupby(by=['at'])['tnmt'].agg({'Count': np.size})
awayTeamCount['Count'] = awayTeamCount['Count'].astype(int)
awayTeamCount = awayTeamCount.sort_values(by = 'Count', ascending=False)
# print(awayTeamCount.head())

winnerCount = results.groupby(by=['winner'])['tnmt'].agg({'Count': np.size})
winnerCount['Count'] = winnerCount['Count'].astype(int)
winnerCount = winnerCount.sort_values(by = 'Count', ascending=False).iloc[1:]
# print(winnerCount.head(20))

#Visualization for Count
# fig = plt.figure(1, figsize=(15,13))
# TopTournament = np.array(tournamentCount.head(15).index)
# TopTournamentData = results[results['tnmt'].isin(TopTournament)]
# ax = sns.countplot(data= TopTournamentData, y ='tnmt',hue = 'era',order=TopTournament)
# plt.title('The Count of Different Tournaments(Top15)', fontsize = 20, weight = 'bold')
# plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);
# plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);
# plt.xlabel('Tournament count', fontsize=16, weight = 'bold', labelpad=10)
# ax.yaxis.label.set_visible(False)


# TopHomeTeam = np.array(homeTeamCount.head(15).index)
# TopHomeTeamData = results[results['ht'].isin(TopHomeTeam)]
# TopAwayTeam = np.array(awayTeamCount.head(15).index)
# TopAwayTeamData = results[results['at'].isin(TopAwayTeam)]
# f, axes = plt.subplots(2, 1, figsize=(14,23))
# plt.sca(axes[0])
# plt.title('The Played Times as Home Team(Top15)', fontsize = 20, weight = 'bold')
# ax = sns.countplot(data=TopHomeTeamData, y='ht', order=TopHomeTeam, hue='era')
# plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);
# plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);
# plt.xlabel('Match Count', fontsize=16, weight = 'bold', labelpad=10)
# ax.yaxis.label.set_visible(False)

# plt.sca(axes[1])
# plt.title('The Played Times as Away Team(Top15)', fontsize = 20, weight = 'bold')
# ax = sns.countplot(data=TopAwayTeamData, y='at', order=TopAwayTeam, hue='era')
# plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);
# plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);
# plt.xlabel('Match Count', fontsize=16, weight = 'bold', labelpad=10)
# ax.yaxis.label.set_visible(False)

# TopWinner = np.array(winnerCount.head(20).index)
# TopWinnerData = results[results['winner'].isin(TopWinner)]
# f, ax = plt.subplots()
# plt.title('The winner teams (Top20)', fontsize=20, weight='bold')
# ax = sns.countplot(data=TopWinnerData, y='winner', order=TopWinner, hue='era')
# plt.setp(ax.get_xticklabels(), fontsize=12, weight='normal', rotation=0)
# plt.setp(ax.get_yticklabels(), fontsize=12, weight='bold', rotation=0)
# plt.xlabel('Match Count', fontsize=16, weight='bold', labelpad=10)
# ax.yaxis.label.set_visible(False)

plt.show()

