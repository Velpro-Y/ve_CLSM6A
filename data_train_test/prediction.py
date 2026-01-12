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
    if nrows == 0:
        raise ValueError("输入文件里没有读到序列。请检查格式：>header 下一行是序列。")

    # 简单校验：长度必须>=201；不足会导致后续取2-mer越界
    for i, s in enumerate(seq):
        if len(s) < 201:
            raise ValueError(f"第{i+1}条序列长度不足201：len={len(s)}。请确保每条序列至少201nt。")
        # 只允许A/C/G/U
        bad = set(s) - set("ACGU")
        if bad:
            raise ValueError(f"第{i+1}条序列包含非法字符 {bad}。请清理或替换为ACGU。")

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

import pandas as pd


D_ENAC = {
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


def df_to_ENAC(df: pd.DataFrame, seq_col: str) -> np.ndarray:

    seqs = df[seq_col].astype(str).str.upper().str.slice(0, 201).str.replace("T", "U", regex=False)
    all_ENAC = []
    for s in seqs:
        s = s[:201]  # 假设你数据已处理好，直接截取
        one_ENAC = []
        for jj in range(200):
            mer2 = s[jj:jj + 2]
            one_ENAC.append(D_ENAC[mer2])
        all_ENAC.append(one_ENAC)
    return np.array(all_ENAC, dtype=np.float32)


@torch.no_grad()
def predict_with_model(model, X_test: np.ndarray, batch_size=64, device="cpu") -> np.ndarray:
    model.eval()
    X = torch.from_numpy(X_test).to(torch.float32).to(device)
    loader = DataLoader(Data.TensorDataset(X), batch_size=batch_size, shuffle=False)

    preds = []
    for (x_batch,) in loader:
        out = model(x_batch)
        preds.append(out[:, 0].detach().cpu().numpy())
    return np.concatenate(preds, axis=0)

import glob
def predict_each_domain_testsets(data_dir: str, models_dir: str, batch_size=64, device="cpu"):
    test_files = sorted(glob.glob(os.path.join(data_dir, "*_test.csv")))
    summary = []

    for test_path in test_files:
        base = os.path.basename(test_path)
        domain = base.replace("_test.csv", "")  # 例如 A549 / liver / brain ...

        ctor_name = f"m_{domain}"
        model_path = os.path.join(models_dir, f"{domain}.pkl")

        # 你说不需要冗余检验，所以这里假定 ctor/model_path 都存在
        model = globals()[ctor_name]()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device)

        df = pd.read_csv(test_path)  # 列：chr,loc,strand,seq,label
        X_test = df_to_ENAC(df, "seq")
        scores = predict_with_model(model, X_test, batch_size=batch_size, device=device)

        df["Score"] = scores
        df["p_label"] = (df["Score"] > 0.5).astype(float)

        out_path = os.path.join(data_dir, f"{domain}_test_pred.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.3f")
        print("Saved:", out_path)

        acc = (df["p_label"] == df["label"].astype(float)).mean()
        print(f"{domain} Accuracy = {acc:.4f}")

        summary.append({"domain": domain, "accuracy": acc, "n": len(df)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(data_dir, "accuracy_summary.csv"), index=False, encoding="utf-8-sig",float_format="%.3f")
    print("Saved:", os.path.join(data_dir, "accuracy_summary.csv"))



if __name__ == "__main__":
    curPath = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curPath, "./")
    models_dir = os.path.join(curPath, "../models")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    print("Device:", device)
    predict_each_domain_testsets(data_dir, models_dir, batch_size=batch_size, device=device)


