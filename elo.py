import torch
from model import DEVICE
import math

def p_win(rating1, rating2):
    return 1/(1+math.pow(10, (rating2-rating1)/400))

def run(data, year, initial_rating={}, k=30, init=1000, trace=0):
    rating = initial_rating.copy()
    season = data.games[data.games.Season==year]
    days = sorted(season.DayNum.unique())
    for day in days:
        games = season[season.DayNum==day]
        for _, game in games[['WTeamID', 'LTeamID']].iterrows():
            winner = game.WTeamID
            loser = game.LTeamID
            winner_rating = rating.get(winner, init)
            loser_rating = rating.get(loser, init)
            prob = p_win(winner_rating, loser_rating)
            rating[winner] = winner_rating + k * (1-prob)
            rating[loser] = loser_rating - k * (1-prob)
            if winner == trace:
                print(f"Beats team at {loser_rating:.2f}: {winner_rating:.2f} -> {rating[winner]:.2f}")
            if loser == trace:
                print(f"Loses to team at {winner_rating:.2f}: {loser_rating:.2f} -> {rating[loser]:.2f}")
    return rating

class EloModel:

    def __init__(self, data, k=30, init=1000, device=DEVICE):
        self.device=device
        self.ratings = {}
        year_ratings = {}
        for year in sorted(data.games.Season.unique()):
            year_ratings = run(data, year, year_ratings, k, init)
            for team, rating in year_ratings.items():
                if (team, year) in data.teamMapping:
                    self.ratings[data.teamMapping[(team, year)]] = rating

    def eval(self):
        pass

    def __call__(self, x):
        teams = x[:,[0,2]]
        probs = [p_win(self.ratings[team1.item()],
                       self.ratings[team2.item()]) for team1, team2 in x[:,[1,3]]]
        return torch.Tensor(probs).to(self.device).reshape([-1,1])
        

def print_elo(ratings, data):
    for team, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{data.all_teams.loc[team].TeamName} ({team}): {rating:.2f}")
