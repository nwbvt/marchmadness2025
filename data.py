import os
import random
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
        self.reverseMapping = {v: k for k,v in self.programMapping.items()}
        mens_teams = pd.read_csv(f'{folder}/MTeams.csv').set_index('TeamID')
        womens_teams = pd.read_csv(f'{folder}/WTeams.csv').set_index('TeamID')
        self.all_teams = pd.concat([mens_teams, womens_teams])
        mens_tourney = pd.read_csv(f'{folder}/MNCAATourneyDetailedResults.csv')
        mens_tourney['League'] = 'M'
        womens_tourney = pd.read_csv(f'{folder}/WNCAATourneyDetailedResults.csv')
        womens_tourney['League'] = 'W'
        self.tourney = pd.concat([mens_tourney, womens_tourney])
        mens_seeds = pd.read_csv('data/MNCAATourneySeeds.csv')
        mens_seeds['League'] = 'M'
        womens_seeds = pd.read_csv('data/WNCAATourneySeeds.csv')
        womens_seeds['League'] = 'W'
        self.seeds = pd.concat([mens_seeds, womens_seeds]).set_index(['Season', 'League', 'TeamID'])
        seeded_tourney = self.tourney.join(self.seeds, on=['Season', 'League', 'WTeamID'])\
                                     .join(self.seeds, on=['Season', 'League', 'LTeamID'], rsuffix='L')
        self.tourney['WSeed'] = seeded_tourney.Seed.map(lambda x: int(x[1:3]))
        self.tourney['LSeed'] = seeded_tourney.SeedL.map(lambda x: int(x[1:3]))
        self.tourney['SeedDiff'] = self.tourney.WSeed - self.tourney.LSeed
        mens_slots = pd.read_csv(f'{folder}/MNCAATourneySlots.csv')
        mens_slots['League'] = 'M'
        womens_slots = pd.read_csv(f'{folder}/WNCAATourneySlots.csv')
        womens_slots['League'] = 'W'
        slots = pd.concat([mens_slots, womens_slots]).set_index(['Season', 'League', 'Slot'])
        seeds_by_slot = pd.concat([mens_seeds, womens_seeds]).set_index(['Season', 'League', 'Seed'])
        self.schedule = slots.\
            join(seeds_by_slot, on=['Season', 'League', 'StrongSeed']).\
            join(seeds_by_slot, on=['Season', 'League', 'WeakSeed'], rsuffix='2')

        
    def gen_dataset(self, games=None, output_stats=False):
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
        if output_stats:
            winner_y = np.concatenate([games[w_stats_columns], games[l_stats_columns]], axis=1)
            loser_y = np.concatenate([games[l_stats_columns], games[w_stats_columns]], axis=1)
        else:
            winner_y = np.ones((n,1))
            loser_y = np.zeros((n,1))
        x_tensor = torch.from_numpy(np.concatenate([winning_matchups, losing_matchups])).double()
        y_tensor = torch.from_numpy(np.concatenate([winner_y, loser_y])).double()
        return TensorDataset(x_tensor, y_tensor)

    def train_test_data(self, train_size=0.9, cache=False, output_stats=False, seed=20250310):
        random.seed(seed)
        train_cache = f"{cache}_train_dataset.pt"
        test_cache = f"{cache}_test_dataset.pt"
        if cache and os.path.isfile(train_cache) and os.path.isfile(test_cache):
            print("Loading cached data")
            train_data = torch.load(train_cache, weights_only=False)
            test_data = torch.load(test_cache, weights_only=False)
        else:
            train_df, test_df = train_test_split(self.games, train_size=train_size)
            print("Generating train dataset")
            train_data = self.gen_dataset(train_df, output_stats=output_stats)
            torch.save(train_data, train_cache)
            print("Generating test dataset")
            test_data = self.gen_dataset(test_df, output_stats=output_stats)
            torch.save(test_data, test_cache)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        return train_loader, test_loader

    def tourney_df(self, year=None, after=None, before=None, league=None):
        df = self.tourney
        if year:
            df = df[df.Season == year]
        elif after:
            df = df[df.Season >= after]
        elif before:
            df = df[df.Season < before]
        if league:
            df = df[df.League == league]
        return df


    def tourney_data(self, year=None, after=None, before=None, league=None, train_size=None):
        df = self.tourney_df(year, after, before, league)
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
        winner_seed = self.seeds.loc[season, :, winner].Seed.max()
        loser_seed = self.seeds.loc[season, :, loser].Seed.max()
        return winner_seed[1:3] > loser_seed[1:3]

    def odds_by_seed_diff(self, year=None, after=None, before=None, league=None):
        tourney_df = self.tourney_df(year, after, before, league)
        seed_diff_counts = tourney_df.SeedDiff.value_counts()
        odds = {0: 0.5}
        for diff in range(1, 16):
            if diff in seed_diff_counts:
                lower_wins = seed_diff_counts[diff]
                higher_wins = seed_diff_counts[-diff]
                odds[diff] = higher_wins/(higher_wins + lower_wins)
                odds[-diff] = lower_wins/(higher_wins + lower_wins)
            else:
                odds[diff] = 1
                odds[-diff] = 0
        return odds
