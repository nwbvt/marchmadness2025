import torch
import torch.nn as nn
import torch.nn.functional as F
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

def train(data, model, loss_fn, optimizer, device=DEVICE, full_loss=True):
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

def test(data, model, loss_fn, device=DEVICE, label="Test"):
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

def gen_submission(model, data, season=2025, device=DEVICE, fname="submission.csv"):
    with open('submission.csv', 'w') as f:
        f.write("ID,Pred\n")
        for league in ('M', 'W'):
            matchups, matchups_tensor = data.all_matchups(season, league)
            predictions, _ = model(matchups_tensor.to(device))
            for (t1, t2), pred in zip(matchups, predictions):
                f.write(f"{season}_{t1.item()}_{t2.item()},{pred.item()}\n")

