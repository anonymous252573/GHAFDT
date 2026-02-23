import torch
import gpytorch
import os
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import copy
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from joint_mds import JointMDS
from scipy.spatial.distance import cdist
from collections import defaultdict
import random
torch.manual_seed(42)
np.random.seed(42)

epochs = 25
prot = 'left'
# prot = 'right'

mice_data_root = r'C:\Users\....\mice_DG\{}_hippo\all_data_interp'.format(prot)
  
h_dataset = 'ADNI'
human_data_root = r'C:\Users\mubar\Downloads\{}_DG'.format(h_dataset)

condis_mice = ['CN', 'AD'] 
condis_human = ['CN_val', 'AD_val', 'CN_test', 'AD_test']

acc_list, auc_list, recall_list, precision_list = [], [], [], []
acc_list_train, auc_list_train, recall_list_train, precision_list_train = [], [], [], []

data_folder_CN_train = os.path.join(mice_data_root, '{}_tangent_data'.format(condis_mice[0]))
data_folder_AD_train = os.path.join(mice_data_root, '{}_tangent_data'.format(condis_mice[1]))

data_folder_CN_val = os.path.join(human_data_root, '{}_tangent_data'.format(condis_human[0]))
data_folder_AD_val = os.path.join(human_data_root, '{}_tangent_data'.format(condis_human[1]))
data_folder_CN_test = os.path.join(human_data_root, '{}_tangent_data'.format(condis_human[2]))
data_folder_AD_test = os.path.join(human_data_root, '{}_tangent_data'.format(condis_human[3]))

def extract_time_map_human(folder):
    time_labels = set()
    for file in os.listdir(folder):
        if not file.endswith('.pkl'):
            continue
        with open(os.path.join(folder, file), 'rb') as f:
            data = pickle.load(f)
            time_labels.update(data.keys())

    # Convert to numeric time values
    def convert_h(label):
        if label == 'bl':
            return 1
        elif label.startswith('m') and label[1:].isdigit():
            return int(label[1:])
        else:
            return None    
    valid_labels = [label for label in time_labels if convert_h(label) is not None]
    time_map = {label: convert_h(label) for label in sorted(valid_labels, key=convert_h)}
    return time_map 

def extract_time_map_mice(folder):
    time_labels = set()
    for file in os.listdir(folder):
        if not file.endswith('.pkl'):
            continue
        with open(os.path.join(folder, file), 'rb') as f:
            data = pickle.load(f)
            time_labels.update(data.keys())

    def convert_m(label):
        if label == '2M':
            return 1
        elif label.endswith('M') and label[:-1].isdigit():
            return int(label[:-1])
        else:
            return None  # Skip invalid labels
    
    valid_labels = [label for label in time_labels if convert_m(label) is not None]
    time_map = {label: convert_m(label) for label in sorted(valid_labels, key=convert_m)}
    return time_map    #bl = 1, m03 = 3, m06= 6, m12 = 12 , etc; returns {'bl': 1, 'm03': 3, ....}  

# **** Let's collect all available timepoints in a dictionary ****
time_map_AD_train = extract_time_map_mice(data_folder_AD_train)
time_map_CN_train = extract_time_map_mice(data_folder_CN_train)

time_map_AD_val = extract_time_map_human(data_folder_AD_val)
time_map_CN_val = extract_time_map_human(data_folder_CN_val)


def load_subject_time_series_human(folder_path, time_map):
    subject_data = {}
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pkl"):
            continue
        if 'bl' not in filename.lower():
            print(f'This file {filename} is missing baseline (bl)')
            continue 
        # Extract subject ID (everything before '_bl')
        try:
            subject_id = filename.split('_bl')[0]
        except IndexError:
            print(f'Unexpected format: {filename}')
            continue
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        sorted_keys = [k for k in time_map if k in data]
        data_ = {k: np.array(data[k]).reshape(-1) for k in sorted_keys}
        subject_data[subject_id] = data_
    return subject_data

def load_subject_time_series_mice(folder_path, time_map):
    subject_data = {}
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pkl"):
            continue

        if '2m' not in filename.lower():
            print(f'This file {filename} is missing baseline (2M)')
            continue 

        # Extract subject ID (everything before '_bl')
        try:
            subject_id = filename.split('_')[0]
        except IndexError:
            print(f'Unexpected format: {filename}')
            continue

        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Use the order from time_map to sort only the keys in data
        sorted_keys = [k for k in time_map if k in data]
        data_ = {k: np.array(data[k]).reshape(-1) for k in sorted_keys}
        subject_data[subject_id] = data_

    return subject_data

subject_CN_dict_train = load_subject_time_series_mice(data_folder_CN_train, time_map_CN_train)
subject_AD_dict_train = load_subject_time_series_mice(data_folder_AD_train, time_map_AD_train)

subject_CN_dict_val = load_subject_time_series_human(data_folder_CN_val, time_map_CN_val)
subject_AD_dict_val = load_subject_time_series_human(data_folder_AD_val, time_map_AD_val)
subject_CN_dict_test = load_subject_time_series_human(data_folder_CN_test, time_map_CN_val)
subject_AD_dict_test = load_subject_time_series_human(data_folder_AD_test, time_map_AD_val)

print("Done Loading Data for Training !!!!")

def get_first_array_dimension(d):
    max_dim = -1
    for subj in d:
        for t in d[subj]:
            current_dim = d[subj][t].shape[-1]
            if current_dim > max_dim:
                max_dim = current_dim

    if max_dim == -1:
        raise ValueError("Dictionary is empty or contains no arrays.")
    return max_dim

def pad_dict_arrays(d, target_dim):
    for subj in d:
        for t in d[subj]:
            arr = d[subj][t]
            if arr.shape[-1] < target_dim:
                pad_width = target_dim - arr.shape[-1]
                d[subj][t] = np.pad(arr, ((0, 0), (0, pad_width)) if arr.ndim == 2 else ((0, pad_width),), mode='constant')
    return d

dim1 = get_first_array_dimension(subject_AD_dict_train)
dim2 = get_first_array_dimension(subject_CN_dict_train)
dim3 = get_first_array_dimension(subject_CN_dict_val)
dim4 = get_first_array_dimension(subject_AD_dict_val)

max_dim = max(dim1, dim2, dim3, dim4)
subject_AD_dict_train = pad_dict_arrays(subject_AD_dict_train, max_dim)
subject_CN_dict_train = pad_dict_arrays(subject_CN_dict_train, max_dim)

subject_AD_dict_val = pad_dict_arrays(subject_AD_dict_val, max_dim)
subject_CN_dict_val = pad_dict_arrays(subject_CN_dict_val, max_dim)
subject_AD_dict_test = pad_dict_arrays(subject_AD_dict_test, max_dim)
subject_CN_dict_test = pad_dict_arrays(subject_CN_dict_test, max_dim)

def prepare_subject_data(group_dict, label, eps=1e-6):
    data = []
    for subject_id, time_dict in group_dict.items():
        times = time_dict.keys()
        subject_seq = []
        for t in times:
            vec = torch.tensor(time_dict[t]).squeeze()
            if vec.ndim != 1:
                raise ValueError(f"Vector at {subject_id} time {t} is not 1D after squeeze: shape={vec.shape}")
            subject_seq.append(vec)
        data.append((subject_seq, label))
    return data

CN_data_train = prepare_subject_data(subject_CN_dict_train, label=0)
AD_data_train = prepare_subject_data(subject_AD_dict_train, label=1)
all_data_train = CN_data_train + AD_data_train 

CN_data_val = prepare_subject_data(subject_CN_dict_val, label=0)
AD_data_val = prepare_subject_data(subject_AD_dict_val, label=1)
all_data_val = CN_data_val + AD_data_val 

CN_data_test = prepare_subject_data(subject_CN_dict_test, label=0)
AD_data_test = prepare_subject_data(subject_AD_dict_test, label=1)
all_data_test = CN_data_test + AD_data_test

class TimeSeriesSubjectDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of [sequence, label]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        return torch.stack(sequence), label  # shape: (T_i, D), label
    
def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)  # (B, T_max, D)
    lengths = [seq.size(0) for seq in sequences] 
    mask = torch.tensor([[i >= l for i in range(padded.size(1))] for l in lengths])
    return padded.float(), torch.tensor(labels), mask
        
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=4, dropout=0.1, max_len=4):
        super(TransformerClassifier, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Embedding(max_len + 1, model_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=1e-4)

        ffn_hidden = 4 * model_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=1e-4)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, x, mask):
        
        B, T, _ = x.size()
        x = self.input_proj(x)

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)  # (1, T+1)
        x = x + self.pos_embedding(positions) 

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        cls_out = x[:, 0, :]  # (B, D)
        out = self.dropout(cls_out)
        out = self.fc_out(out) 
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# domain alignment and classification
def extract_samples(all_data):
    X = []
    meta = []
    for subj_idx, (subject_seq, label) in enumerate(all_data):
        for t_idx, vec in enumerate(subject_seq):
            X.append(vec.numpy())
            meta.append((subj_idx, t_idx, label))
    return np.stack(X), meta
X_mouse, meta_mouse = extract_samples(all_data_train)
X_human_val, meta_human_val = extract_samples(all_data_val)
X_human_test, meta_human_test = extract_samples(all_data_test)

D1 = cdist(X_mouse, X_mouse, metric="euclidean")
D2 = cdist(X_human_val, X_human_val, metric="euclidean")

D3 = cdist(X_human_test, X_human_val)

print(D1.shape, D2.shape, D3.shape)

# Barycentric projection
W = 1.0 / (D3 + 1e-8)
W /= W.sum(axis=1, keepdims=True)

for n_comp in range(150, 255, 5):

    acc_list, auc_list, recall_list, precision_list = [], [], [], []
    acc_list_train, auc_list_train, recall_list_train, precision_list_train = [], [], [], []

    print('\n',f'*******************************************************')
    print( f" ********* NOW RUNNING FOR {n_comp} COMPONENTS ************")   
    print('*********************************************************')

    JMDS = JointMDS(n_components=n_comp, dissimilarity="precomputed")
    Z_mouse, Z_human_val, P = JMDS.fit_transform(torch.tensor(D1), torch.tensor(D2))   
    Z_human_test  = torch.tensor(W) @ Z_human_val

    def rebuild_subject_data(Z, meta):
        subject_dict = defaultdict(list)
        label_dict = {}

        for z, (subj_idx, t_idx, label) in zip(Z, meta):
            subject_dict[subj_idx].append((t_idx, z / (torch.norm(z) + 1e-8)))
            label_dict[subj_idx] = label
        data = []
        for subj_idx in subject_dict:
            seq = [z for _, z in subject_dict[subj_idx]]
            data.append((seq, label_dict[subj_idx]))
        return data

    mouse_data_JMDS = rebuild_subject_data(Z_mouse, meta_mouse)
    human_data_JMDS_val = rebuild_subject_data(Z_human_val, meta_human_val)
    human_data_JMDS_test = rebuild_subject_data(Z_human_test, meta_human_test)

    all_data_train = mouse_data_JMDS
    data_val = human_data_JMDS_val
    data_test = human_data_JMDS_test    

    train_dataset = TimeSeriesSubjectDataset(all_data_train)
    val_dataset = TimeSeriesSubjectDataset(data_val)
    test_dataset = TimeSeriesSubjectDataset(data_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    input_dim = all_data_train[0][0][0].shape[0]
    model = TransformerClassifier(input_dim=input_dim, model_dim=128, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    def train_one_epoch(model, dataloader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_labels = []
        all_probs = []
        all_preds = []
        
        for x, y, mask in dataloader:
            
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()

            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / total
        try:
            accuracy = balanced_accuracy_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_probs)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
        except ValueError:
            auc = float('nan')

        return avg_loss, accuracy, auc, precision, recall

    @torch.no_grad()
    def evaluate(model, dataloader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_labels = []
        all_probs = []
        all_preds = []

        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            logits = model(x, mask)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            # For metrics
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / total

        try:
            accuracy = balanced_accuracy_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_probs)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
        except ValueError:
            auc = float('nan')
            precision = float('nan')
            recall = float('nan')

        return avg_loss, accuracy, auc, precision, recall

    best_val_acc = 0
    best_val_auc = 0
    best_val_rec = 0
    best_metrics = {}
    train_metrics = {}
    val_metrics = {}

    for epoch in range(epochs):
        # Train and evaluate
        train_loss, train_acc, train_auc, train_preci, train_rec = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc, val_preci, val_rec = evaluate(model, val_loader, criterion)
        test_loss, test_acc, test_auc, test_preci, test_rec = evaluate(model, test_loader, criterion)
        
        if (val_acc >= best_val_acc) and (val_auc >= best_val_auc) and (val_rec >= best_val_rec):

            best_val_acc = val_acc
            best_val_auc = val_auc
            best_val_rec = val_rec

            # Save best metrics
            best_metrics = {
                'epoch': epoch + 1,
                'acc': test_acc,
                'auc': test_auc,
                'loss': test_loss,
                'prec': test_preci,
                'rec': test_rec
            }

            train_metrics = {
                'epoch': epoch + 1,
                'acc': train_acc,
                'auc': train_auc,
                'loss': train_loss,
                'prec': train_preci,
                'rec': train_rec
            }

            val_metrics = {
                'epoch': epoch + 1,
                'acc': val_acc,
                'auc': val_auc,
                'loss': val_loss,
                'prec': val_preci,
                'rec': val_rec
            }

    acc_list.append(best_metrics['acc'])
    auc_list.append(best_metrics['auc'])
    recall_list.append(best_metrics['rec'])
    precision_list.append(best_metrics['prec'])

    acc_arr = np.array(acc_list)
    auc_arr = np.array(auc_list)
    rec_arr = np.array(recall_list)
    prec_arr = np.array(precision_list)

    # =====================
    acc_list_train.append(train_metrics['acc'])
    auc_list_train.append(train_metrics['auc'])
    recall_list_train.append(train_metrics['rec'])
    precision_list_train.append(train_metrics['prec'])

    acc_arr = np.array(acc_list)
    auc_arr = np.array(auc_list)
    rec_arr = np.array(recall_list)
    prec_arr = np.array(precision_list)

    acc_arr_train = np.array(acc_list_train)
    auc_arr_train = np.array(auc_list_train)
    rec_arr_train = np.array(recall_list_train)
    prec_arr_train = np.array(precision_list_train)

    print('\n'
        f"Train perf.-> "
        f"Acc: {acc_arr_train.mean():.4f}, "
        f"AUC: {auc_arr_train.mean():.4f}, "
        f"Recall: {rec_arr_train.mean():.4f}, "
        f"Precision: {prec_arr_train.mean():.4f}"
    )

    print(
        f"Test perf. -> "
        f"Acc: {acc_arr.mean():.4f}, "
        f"AUC: {auc_arr.mean():.4f}, "
        f"Recall: {rec_arr.mean():.4f}, "
        f"Precision: {prec_arr.mean():.4f}"
    )
