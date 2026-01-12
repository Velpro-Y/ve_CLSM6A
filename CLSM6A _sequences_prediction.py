import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")
import sys


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
from pandas import DataFrame


class m_A549(nn.Module):
    def __init__(self):
        super(m_A549, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_brain(nn.Module):
    def __init__(self):
        super(m_brain, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_CD8T(nn.Module):
    def __init__(self):
        super(m_CD8T, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HCT116(nn.Module):
    def __init__(self):
        super(m_HCT116, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 64),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(64, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HEK293(nn.Module):
    def __init__(self):
        super(m_HEK293, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HEK293T(nn.Module):
    def __init__(self):
        super(m_HEK293T, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(3200, 128),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(128, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HeLa(nn.Module):
    def __init__(self):
        super(m_HeLa, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_HepG2(nn.Module):
    def __init__(self):
        super(m_HepG2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(3200, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_kidney(nn.Module):
    def __init__(self):
        super(m_kidney, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(400, 128),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(128, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_liver(nn.Module):
    def __init__(self):
        super(m_liver, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

class m_MOLM13(nn.Module):
    def __init__(self):
        super(m_MOLM13, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(nn.Linear(1600, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2)
                                )
        self.classifier = nn.Sequential(nn.Linear(256, 1),
                                nn.Sigmoid(),
                                )

    def forward(self,x):

        x = torch.transpose(x,1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.classifier(x)
        return out

def read_test_txt(data_path):
    seq_name = []
    seq = []

    with open(data_path, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            if line.startswith(">"):
                line = line.strip()
                seq_name.append(line)
            elif line.strip() == "":
                continue
            else:
                # 只取前201nt，并统一大写；把DNA的T替换成U（避免字典KeyError）
                line = line.strip().upper()[:201].replace("T", "U")
                seq.append(line)

    return seq_name, seq


def read_test_txt_to_ENAC(data_path):
    seq_name, seq = read_test_txt(data_path)
    clean_seq_name, clean_seq = [], []
    dropped = 0
    for name, s in zip(seq_name, seq):
        bad = set(s) - set("ACGU")
        if bad:
            dropped += 1
            continue
        clean_seq_name.append(name)
        clean_seq.append(s)

    print(f"Dropped {dropped} sequences due to illegal chars (e.g., N).")

    seq_name, seq = clean_seq_name, clean_seq

    d = {
        "AA": [1, 0, 0, 0],
        "AC": [0.5, 0.5, 0, 0],
        "AG": [0.5, 0, 0.5, 0],
        "AU": [0.5, 0, 0, 0.5],
        "CA": [0.5, 0.5, 0, 0],
        "CC": [0, 1, 0, 0],
        "CG": [0, 0.5, 0.5, 0],
        "CU": [0, 0.5, 0, 0.5],
        "GA": [0.5, 0, 0.5, 0],
        "GC": [0, 0.5, 0.5, 0],
        "GG": [0, 0, 1, 0],
        "GU": [0, 0, 0.5, 0.5],
        "UA": [0.5, 0, 0, 0.5],
        "UC": [0, 0.5, 0, 0.5],
        "UG": [0, 0, 0.5, 0.5],
        "UU": [0, 0, 0, 1],
    }

    nrows = len(seq)

    all_ENAC = []
    for i in range(nrows):
        one_seq = seq[i]
        one_ENAC = []
        # 201nt -> 200个2-mer
        for jj in range(200):
            mer2 = one_seq[jj:jj + 2]
            one_ENAC.append(d[mer2])
        all_ENAC.append(one_ENAC)

    return seq_name, seq, np.array(all_ENAC, dtype=np.float32)



def predict_with_model(model, X_test, batch_size=32, device="cpu"):
    model.eval()
    X = torch.from_numpy(X_test).to(torch.float32).to(device)
    test_dataset = Data.TensorDataset(X)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pred_all = []
    with torch.no_grad():
        for (x_batch,) in test_iter:
            out = model(x_batch)
            pred_all.append(out[:, 0].detach().cpu().numpy())
    return np.concatenate(pred_all, axis=0)


def main_all_models(example_dir, jobid, models_dir, batch_size=32, device="cpu"):

    in_file = os.path.join(example_dir, f"{jobid}.txt")
    seq_name, seq, X_test = read_test_txt_to_ENAC(in_file)

    model_list = ["A549", "CD8T", "HCT116", "HEK293", "HEK293T", "HeLa",
                  "HepG2", "MOLM13", "brain", "kidney", "liver"]

    scores = {}  # model_name -> (N,)
    for mname in model_list:
        ctor_name = f"m_{mname}"

        model = globals()[ctor_name]()
        weight_path = os.path.join(models_dir, f"{mname}.pkl")

        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.to(device)
        scores[mname] = predict_with_model(model, X_test, batch_size=batch_size, device=device)

    df = DataFrame({"Name": seq_name, "Sequence": seq})
    for mname in model_list:
        df[f"Score_{mname}"] = scores[mname]

    score_cols = [f"Score_{m}" for m in model_list]
    df["MaxScore"] = df[score_cols].max(axis=1)
    df["MaxModel"] = df[score_cols].idxmax(axis=1).str.replace("Score_", "", regex=False)

    out_csv = os.path.join(example_dir, f"{jobid}_results_all_models.csv")
    df.to_csv(out_csv, index=False, encoding="gbk", float_format="%.6f")
    print("Saved:", out_csv)


if __name__ == "__main__":

    jobid = "01prediction"
    batch_size = 32


    curPath = os.path.dirname(os.path.realpath(__file__))
    example_dir = os.path.join(curPath, "example")
    models_dir = os.path.join(curPath, "models")


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    main_all_models(example_dir=example_dir, jobid=jobid, models_dir=models_dir,
                    batch_size=batch_size, device=device)


