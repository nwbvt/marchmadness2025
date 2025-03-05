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
    def __init__(self, folder="data", batch_size=500):
        self.batch_size=batch_size
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
        mens_tourney = pd.read_csv(f'{folder}/MNCAATourneyDetailedResults.csv')
        mens_tourney['League'] = 'M'
        womens_tourney = pd.read_csv(f'{folder}/WNCAATourneyDetailedResults.csv')
        womens_tourney['League'] = 'W'
        self.tourney = pd.concat([mens_tourney, womens_tourney])
        mens_seeds = pd.read_csv('data/MNCAATourneySeeds.csv')
        womens_seeds = pd.read_csv('data/WNCAATourneySeeds.csv')
        self.seeds = pd.concat([mens_seeds, womens_seeds]).set_index(['Season', 'TeamID'])
        
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

    def train_test_data(self, train_size=0.9, use_cache=True):
        train_cache = "train_dataset.pt"
        test_cache = "test_dataset.pt"
        if use_cache and os.path.isfile(train_cache) and os.path.isfile(test_cache):
            print("Loading cached data")
            train_data = torch.load(train_cache, weights_only=False)
            test_data = torch.load(test_cache, weights_only=False)
        else:
            train_df, test_df = train_test_split(self.games)
            print("Generating train dataset")
            train_data = self.gen_dataset(train_df)
            torch.save(train_data, train_cache)
            print("Generating test dataset")
            test_data = self.gen_dataset(test_df)
            torch.save(test_data, test_cache)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        return train_loader, test_loader

    def tourney_data(self, year=None, after=None, before=None, league=None, train_size=None):
        df = self.tourney
        if year:
            df = df[df.Season == year]
        elif after:
            df = df[df.Season >= after]
        elif before:
            df = df[df.Season < before]
        if league:
            df = df[df.League == league]
        if train_size is None:
            return DataLoader(self.gen_dataset(df), batch_size=self.batch_size)
        else:
            train, test = train_test_split(df, train_size)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=self.batch_size)
            return train_loader, test_loader

    def all_matchups(self, season, league):
        teams_to_test = sorted(self.teams[(self.teams.Season==season) & (self.teams.League==league)].TeamID.values)
        matchups = [(t1, t2) for t1 in teams_to_test for t2 in teams_to_test if t1 < t2]
        matchups_tensor = torch.Tensor(np.array(
            [[self.programMapping[t1], self.teamMapping[(t1, season)],
              self.programMapping[t2], self.teamMapping[(t2, season)],
              season, 140, league == 'M'] for (t1, t2) in matchups])).int()
        return matchups, matchups_tensor

    def upset(self, season, winner, loser):
        winner_seed = self.seeds.loc[season, winner].Seed
        loser_seed = self.seeds.loc[season, loser].Seed
        return winner_seed[1:3] > loser_seed[1:3]
