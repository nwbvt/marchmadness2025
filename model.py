import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import STATS_COLUMNS

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class MatchupLayer(nn.Module):
    def __init__(self, program_embedding, team_embedding, num_programs, num_teams):
        super(MatchupLayer, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, team_embedding)
        self.program_embedding = nn.Embedding(num_programs, program_embedding)
        self.double()

    def forward(self, x):
        program = self.program_embedding(x[:,0].int())
        team = self.team_embedding(x[:,1].int())
        opponent_program = self.program_embedding(x[:,2].int())
        opponent = self.team_embedding(x[:,3].int())
        matchup = torch.cat([program, team, opponent_program, opponent, x[:,4:]], axis=1)
        return matchup

    def freeze(self):
        for param in self.team_embedding.parameters():
            param.requires_grad=False
        for param in self.program_embedding.parameters():
            param.requires_grad=False

    def unfreeze(self):
        for param in self.team_embedding.parameters():
            param.requires_grad=True
        for param in self.program_embedding.parameters():
            param.requires_grad=True

class StatsModel(nn.Module):
    def __init__(self, program_embedding, team_embedding, num_programs, num_teams, model_sizes, dropout):
        super(StatsModel, self).__init__()
        hid1, hid2 = model_sizes
        self.matchup = MatchupLayer(program_embedding, team_embedding, num_programs, num_teams)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2*self.matchup.team_embedding.embedding_dim +\
                             2*self.matchup.program_embedding.embedding_dim + 3, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, 2*len(STATS_COLUMNS))
        self.double()

    def forward(self, x):
        matchup = self.dropout(self.matchup(x))
        hidden1 = self.dropout(F.relu(self.fc1(matchup)))
        hidden2 = self.dropout(F.relu(self.fc2(hidden1)))
        return self.fc3(hidden2)

class Model(nn.Module):
    def __init__(self, matchup, model_sizes, dropout):
        super(Model, self).__init__()
        hid1, hid2 = model_sizes
        self.matchup = matchup
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2*matchup.team_embedding.embedding_dim +\
                             2*matchup.program_embedding.embedding_dim + 3, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, 1)
        self.double()

    def forward(self, x):
        matchup = self.dropout(self.matchup(x))
        hidden1 = self.dropout(F.relu(self.fc1(matchup)))
        hidden2 = self.dropout(F.relu(self.fc2(hidden1)))
        return F.sigmoid(self.fc3(hidden2))

def train_epoch(data, model, optimizer, device=DEVICE):
    loss_fn = nn.MSELoss()
    size = len(data.dataset)
    model.train()
    for batch, (x, y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>6d}/{size:>6d}]", end="\r")

def test(data, model, device=DEVICE):
    loss_fn = nn.MSELoss()
    size = len(data.dataset)
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).item()*len(x)
    return loss / size

def test_accuracy(data, model, device=DEVICE):
    size = len(data.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            correct += ((pred >= 0.5) == (y == 1)).type(torch.float).sum().item()
    return correct / size

def print_results(data, model, label, device=DEVICE):
    loss = test(data, model, device)
    accuracy = test_accuracy(data, model, device)
    print(f"{label}: Accuracy={accuracy*100:3.2f}, Loss={loss:>8f}")

def train(train_data, test_data, model, name="model", learning_rate=0.001,
          device=DEVICE, full_loss=True, max_epochs=100, streak=5, use_cache=False):
    fname=f"{name}.pth"
    if use_cache and os.path.isfile(fname):
        print("Loading from cache")
        model.load_state_dict(torch.load(fname))
        return
    torch.save(model.state_dict(), fname)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = test(test_data, model, device)
    curr_streak = 0
    for i in range(max_epochs):
        train_epoch(train_data, model, optimizer, device)
        train_loss=test(train_data, model, device)
        loss = test(test_data, model, device)
        print(f"Epoch {i:3d}: Train Loss={train_loss:3.8f}, Test Loss={loss:3.8f}")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), fname)
            curr_streak = 0
        else:
            curr_streak += 1
            if curr_streak >= streak:
                break
    print(f"Best Loss: {best_loss:>8f}")
    model.load_state_dict(torch.load(fname))
    return

def feature_eval(model, data, device=DEVICE):
    model.eval()
    team_grads = torch.zeros(model.matchup.team_embedding.embedding_dim).to(device)
    program_grads = torch.zeros(model.matchup.program_embedding.embedding_dim).to(device)
    stats_grads = torch.zeros(3).to(device)
    size = len(data.dataset)
    for batch, (x, y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        x.requires_grad = True
        pred_result = model(x)
        team_grads += torch.autograd.grad(model(x)[1].mean(), model.matchup.team_embedding.parameters())[0].sum(axis=0)
        program_grads += torch.autograd.grad(model(x)[1].mean(), model.matchup.program_embedding.parameters())[0].sum(axis=0)
        stats_grads += torch.autograd.grad(model(x)[1].mean(), x)[0].sum(axis=0)[4:]
    return program_grads/size, team_grads/size, stats_grads

def model_odds(data, season, league, model, device=DEVICE):
    matchups, matchups_tensor = data.all_matchups(season, league)
    predictions = model(matchups_tensor.to(device))
    return {(int(t1), int(t2)): pred.item() for ((t1, t2), pred) in zip(matchups, predictions)}

def gen_submission(model, data, season=2025, device=DEVICE, fname="submission.csv"):
    with open(fname, 'w') as f:
        f.write("ID,Pred\n")
        for league in ('M', 'W'):
            for (t1, t2), pred in model_odds(data, season, league, model, device).items():
                f.write(f"{season}_{t1}_{t2},{pred}\n")

class ModeratedModel:
    def __init__(self, model, weight, device=DEVICE):
        self.model = model
        self.weight = weight
        self.device=device

    def eval(self):
        pass

    def __call__(self, x):
        model_score = self.model(x)
        neutral = torch.Tensor(np.array([0.5]*len(model_score)).reshape((-1,1))).to(self.device)
        return model_score * self.weight + neutral * (1-self.weight)

class SeedModel:
    def __init__(self, dataset, year=None, after=None, before=None, league=None, device=DEVICE):
        self.odds = dataset.odds_by_seed_diff(year, after, before, league)
        self.seeds = dataset.seeds
        self.device=device
        self.programs = dataset.programs
        self.program_mapping = dataset.reverseMapping

    def eval(self):
        pass

    def seed(self, season, league, team):
        if (season, league, self.program_mapping[team]) in self.seeds.index:
            return int(self.seeds.loc[season, league, self.program_mapping[team]].Seed[1:3])
        else:
            return -1

    def win_odds(self, team1, team2):
        if team1 == -1:
            if team2 == -1:
                return 0.5
            return 0
        if team2 == -1:
            return 1
        return self.odds[team1-team2]
        
    
    def __call__(self, x):
        team_1_seed = [self.seed(s.item(), 'M' if mens else 'W', t.item()) for s,t,mens in x[:,[4,0,6]]]
        team_2_seed = [self.seed(s.item(), 'M' if mens else 'W', t.item()) for s,t,mens in x[:,[4,2,6]]]
        results = torch.Tensor([self.win_odds(t1, t2) for t2, t1 in
                                zip(team_1_seed, team_2_seed)]).to(self.device).reshape([-1,1])
        return results

class HybridModel(object):
    def __init__(self, models, weights, device=DEVICE):
        self.models = models
        self.weights = weights
        self.device = device

    def eval(self):
        pass

    def __call__(self, x):
        results = torch.zeros(len(x)).reshape([-1,1]).to(self.device)
        for model, weight in zip(self.models, self.weights):
            result = model(x)
            results += weight * result
        return results

def gen_bracket(data, season, league, model):
    odds = model_odds(data, season, league, model)
    schedule = data.schedule.loc[season, league, :].copy()
    schedule.insert(len(schedule.columns), 'Winner', -1)
    schedule.insert(len(schedule.columns), 'P', -1.0)
    i=0
    while sum(schedule.Winner < 0) and i <= 10:
        i+=1
        games = schedule[(schedule.Winner < 0) & schedule.TeamID.notna() & schedule.TeamID2.notna()][['TeamID', 'TeamID2']]
        for slot, t1, t2 in games.itertuples():
            if t1 > t2:
                t1, t2 = t2, t1
            p = odds[(t1, t2)]
            schedule.loc[slot, 'P'] = p
            winner = t1 if p > 0.5 else t2
            schedule.loc[slot, 'Winner'] = winner
            schedule.loc[schedule.StrongSeed == slot, 'TeamID'] = winner
            schedule.loc[schedule.WeakSeed == slot, 'TeamID2'] = winner
    return schedule
