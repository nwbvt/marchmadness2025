import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import STATS_COLUMNS

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self, embedding_sizes, model_sizes, dropout, dataset):
        super(Model, self).__init__()
        p_embedding_size, t_embedding_size = embedding_sizes
        hid1, hid2 = model_sizes
        self.team_embedding = nn.Embedding(len(dataset.teams), p_embedding_size)
        self.program_embedding = nn.Embedding(len(dataset.programs), t_embedding_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2*p_embedding_size+2*t_embedding_size+3, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.stats_fc = nn.Linear(hid2, 2*len(STATS_COLUMNS))
        self.result_fc = nn.Linear(hid2, 1)
        self.double()

    def forward(self, x):
        program = self.program_embedding(x[:,0].int())
        team = self.team_embedding(x[:,1].int())
        opponent_program = self.program_embedding(x[:,2].int())
        opponent = self.team_embedding(x[:,3].int())
        matchup = self.dropout1(torch.cat([program, team, opponent_program, opponent, x[:,4:]], axis=1))
        hidden1 = self.dropout2(F.relu(self.fc1(matchup)))
        hidden2 = self.dropout3(F.relu(self.fc2(hidden1)))
        stats = self.stats_fc(hidden2)
        result = F.sigmoid(self.result_fc(hidden2))
        return result, stats

def train_epoch(data, model, optimizer, device=DEVICE, full_loss=True):
    loss_fn = nn.MSELoss()
    size = len(data.dataset)
    model.train()
    for batch, (x, y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        pred_result, pred_stats = model(x)
        actual_result = y[:,0].double().reshape((-1,1))
        actual_stats = y[:,1:].double()
        result_loss = loss_fn(pred_result, actual_result)
        stats_loss = loss_fn(pred_stats, actual_stats)
        if full_loss:
            (stats_loss + result_loss).backward()
        else:
            result_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            result_loss, current = result_loss.item(), (batch + 1) * len(x)
            print(f"result loss: {result_loss:>7f} [{current:>6d}/{size:>6d}]", end="\r")

def test(data, model, device=DEVICE, label="Test"):
    loss_fn = nn.MSELoss()
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    stats_loss, result_loss, correct = 0, 0, 0
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            pred_result, pred_stats = model(x)
            actual_result = y[:,0].double().reshape((-1,1))
            actual_stats = y[:,1:].double()
            result_loss += loss_fn(pred_result, actual_result).item()*len(x)
            stats_loss += loss_fn(pred_stats, actual_stats).item()*len(x)
            correct += ((pred_result >= 0.5) == (actual_result == 1)).type(torch.float).sum().item()
    stats_loss /= size
    result_loss /= size
    correct /= size
    print(f"{label}: Accuracy: {(100*correct):>0.2f}%, Stats loss: {stats_loss:>8f} Result loss: {result_loss:>8f}")
    return result_loss

def train(train_data, test_data, model, learning_rate, 
          device=DEVICE, full_loss=True,
          max_epochs=100, streak=5, checkpoint="checkpoint.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = test(test_data, model, device, label="Initial")
    curr_streak = 0
    for i in range(max_epochs):
        print(f"Epoch {i}")
        train_epoch(train_data, model, optimizer, device, full_loss)
        test(train_data, model, device, label="Train")
        loss = test(test_data, model, device, label="Test")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), checkpoint)
            curr_streak = 0
        else:
            curr_streak += 1
            if curr_streak >= streak:
                break
    print(f"Best Loss: {best_loss:>8f}")
    model.load_state_dict(torch.load(checkpoint))
    return

def feature_eval(model, data, device=DEVICE):
    model.eval()
    team_grads = torch.zeros(model.team_embedding.embedding_dim).to(device)
    program_grads = torch.zeros(model.program_embedding.embedding_dim).to(device)
    stats_grads = torch.zeros(3).to(device)
    size = len(data.dataset)
    for batch, (x, y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        x.requires_grad = True
        _, pred_result = model(x)
        team_grads += torch.autograd.grad(model(x)[1].mean(), model.team_embedding.parameters())[0].sum(axis=0)
        program_grads += torch.autograd.grad(model(x)[1].mean(), model.program_embedding.parameters())[0].sum(axis=0)
        stats_grads += torch.autograd.grad(model(x)[1].mean(), x)[0].sum(axis=0)[4:]
    return program_grads/size, team_grads/size, stats_grads

def model_odds(data, season, league, model, device=DEVICE):
    matchups, matchups_tensor = data.all_matchups(season, league)
    predictions, _ = model(matchups_tensor.to(device))
    return {(int(t1), int(t2)): pred.item() for ((t1, t2), pred) in zip(matchups, predictions)}

def gen_submission(model, data, season=2025, device=DEVICE, fname="submission.csv"):
    with open('submission.csv', 'w') as f:
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
        scores, model_score = self.model(x)
        neutral = torch.Tensor(np.array([0.5]*len(model_score)).reshape((-1,1))).to(self.device)
        return scores, model_score * self.weight + neutral * (1-self.weight)

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
        stats = torch.zeros((len(x), len(STATS_COLUMNS*2))).to(self.device)
        results = torch.Tensor([self.win_odds(t1, t2) for t2, t1 in
                                zip(team_1_seed, team_2_seed)]).to(self.device).reshape([-1,1])
        return results, stats

class HybridModel(object):
    def __init__(self, models, weights, device=DEVICE):
        self.models = models
        self.weights = weights
        self.device = device

    def eval(self):
        pass

    def __call__(self, x):
        results = torch.zeros(len(x)).reshape([-1,1]).to(self.device)
        stats = torch.zeros((len(x), len(STATS_COLUMNS*2))).to(self.device)
        for model, weight in zip(self.models, self.weights):
            result, stats = model(x)
            results += weight * result
            stats += weight * stats
        return results, stats

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
