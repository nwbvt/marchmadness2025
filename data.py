import os
import pandas as pd
from dataclasses import dataclass
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

STATS_COLUMNS = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
                 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

class Data:
    def __init__(self, folder="data"):
        mens = pd.read_csv(f"{folder}/MRegularSeasonDetailedResults.csv")
        mens['League'] = 'M'
        womens = pd.read_csv(f"{folder}/WRegularSeasonDetailedResults.csv")
        womens['League'] = 'W'
        self.games = pd.concat([mens, womens])
        self.teams = pd.concat([self.games[['WTeamID', 'Season', 'League']].rename(columns={'WTeamID': 'TeamID'}),
                                self.games[['LTeamID', 'Season', 'League']].rename(columns={'LTeamID': 'TeamID'})]
                               ).drop_duplicates().reset_index()
        self.programs = self.teams.TeamID.drop_duplicates().reset_index()
        self.teamMapping = {(x.TeamID, x.Season): x.Index for x in self.teams.itertuples()}
        self.programMapping = {x.TeamID: x.Index for x in self.programs.itertuples()}
        mens_teams = pd.read_csv(f'{folder}/MTeams.csv').set_index('TeamID')
        womens_teams = pd.read_csv(f'{folder}/WTeams.csv').set_index('TeamID')
        self.all_teams = pd.concat([mens_teams, womens_teams])
        
    def gen_dataset(self, games=None):
        if games is None:
            games = self.games
        w_stats_columns = [f"W{stat}" for stat in STATS_COLUMNS]
        l_stats_columns = [f"L{stat}" for stat in STATS_COLUMNS]
        n = len(games)
        winning_team = games.apply(lambda x: self.teamMapping[(x.WTeamID, x.Season)], axis=1)
        losing_team = games.apply(lambda x: self.teamMapping[(x.LTeamID, x.Season)], axis=1)
        winning_program = games.apply(lambda x: self.programMapping[x.WTeamID], axis=1)
        losing_program = games.apply(lambda x: self.programMapping[x.LTeamID], axis=1)
        winning_matchups = np.stack([winning_program, winning_team,
                                     losing_program, losing_team,
                                     games.Season, games.DayNum, games.League == 'M'], axis=1)
        losing_matchups = np.stack([losing_program, losing_team,
                                    winning_program, winning_team, 
                                    games.Season, games.DayNum, games.League == 'M'], axis=1)
        winner_y = np.concatenate([np.ones((n, 1)), games[w_stats_columns], games[l_stats_columns]], axis=1)
        loser_y = np.concatenate([np.zeros((n, 1)), games[l_stats_columns], games[w_stats_columns]], axis=1)
        x_tensor = torch.from_numpy(np.concatenate([winning_matchups, losing_matchups])).double()
        y_tensor = torch.from_numpy(np.concatenate([winner_y, loser_y])).double()
        return TensorDataset(x_tensor, y_tensor)

    def train_test_data(self, train_size=0.9, batch_size=500, use_cache=True):
        train_cache = "train_dataset.pt"
        test_cache = "test_dataset.pt"
        if use_cache and os.path.isfile(train_cache) and os.path.isfile(test_cache):
            train_data = torch.load(train_cache, weights_only=False)
            test_data = torch.load(test_cache, weights_only=False)
        else:
            train_df, test_df = train_test_split(self.games)
            train_data = self.gen_dataset(train_df)
            test_data = self.gen_dataset(test_df)
        return DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(test_data, batch_size=batch_size)

