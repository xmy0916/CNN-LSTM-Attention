from dataloader import StockDataset
from torch.utils.data import DataLoader
from model import *
import torch.nn as nn
import os

if not os.path.exists("result_picture"):
    os.makedirs("result_picture")

if not os.path.exists("best_model"):
    os.makedirs("best_model")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="which model", type=str, default="Base")
    parser.add_argument("--epochs", "-e", help="total epoch", type=int, default=100)
    parser.add_argument("--step", "-s", help="print step", type=int, default=10)
    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    train_data = StockDataset('dataset/data.csv', 5, is_test=False)
    test_data = StockDataset('dataset/data.csv', 5, is_test=True)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

    model_dict = {
        "Base": CNNLSTMModel,
        "SE": CNNLSTMModel_SE,
        "ECA": CNNLSTMModel_ECA,
        "CBAM": CNNLSTMModel_CBAM,
        "HW": CNNLSTMModel_HW
    }

    model = model_dict[args.model]()

    print(model)
    print(f"training model is {args.model}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    min_loss = float("inf")
    for epoch in range(args.epochs):
        print(f'epoch:{epoch}')
        running_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % args.step == 0:
                with torch.no_grad():
                    mse_loss = 0.0
                    for data, label in test_loader:
                        out = model(data)
                        loss = criterion(out, label)
                        mse_loss += loss.item()
                    if mse_loss / len(test_loader) < min_loss:
                        torch.save(model.state_dict(), f"best_model/{args.model}_best.pth")
                        print("save_best")
                        min_loss = mse_loss / len(test_loader)
                    print(f"step:{step}, test loss:{mse_loss / len(test_loader)}")

    print("done")

if __name__ in '__main__':
    train()
