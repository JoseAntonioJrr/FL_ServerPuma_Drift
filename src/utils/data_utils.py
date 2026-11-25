# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import torch
from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    """
    Dataset wrapper que armazena tensores brutos na RAM (ex: 32x32)
    e aplica transformações (Resize 224x224, Normalização) sob demanda (on-the-fly)
    durante o __getitem__, evitando a expansão de RAM no sistema.
    """
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_i = self.X[idx]
        if self.transform is not None:
            x_i = self.transform(x_i)
        return x_i, self.y[idx]


def _dataset_root():
    """Raiz dos datasets (padrão: ../dataset relativo a src/system). Sobrescrevível via PUMA_DATASET_ROOT."""
    return os.environ.get('PUMA_DATASET_ROOT', os.path.join('..', 'dataset'))


_STRICT_CACHE = {}


def _strict_phase_enabled(dataset):
    """Modo estrito: fase ausente gera erro em vez de cair silenciosamente em {idx}.npz.

    Ativado por PUMA_STRICT_PHASE=1 ou por ``"strict_phase": true`` no config.json do dataset
    (gravado pelo gerador ASTRA). Para datasets legados o comportamento antigo é mantido.
    """
    env = os.environ.get('PUMA_STRICT_PHASE')
    if env is not None:
        return env.strip().lower() in ('1', 'true', 'yes')
    root = _dataset_root()
    cache_key = (root, dataset)
    if cache_key not in _STRICT_CACHE:
        strict = False
        cfg_path = os.path.join(root, dataset, 'config.json')
        try:
            import json
            with open(cfg_path, 'r') as f:
                strict = bool(json.load(f).get('strict_phase', False))
        except (OSError, ValueError):
            strict = False
        _STRICT_CACHE[cache_key] = strict
    return _STRICT_CACHE[cache_key]


def _resolve_file(data_dir, dataset, idx, phase):
    if phase:
        phase_file = os.path.join(data_dir, f"{idx}_{phase}.npz")
        if os.path.exists(phase_file):
            return phase_file
        if _strict_phase_enabled(dataset):
            raise FileNotFoundError(
                f"[STRICT PHASE] Arquivo de fase ausente: {phase_file} (sem fallback para {idx}.npz)")
    return os.path.join(data_dir, f"{idx}.npz")


def read_data(dataset, idx, is_train=True, phase=None):
    if is_train:
        train_data_dir = os.path.join(_dataset_root(), dataset, 'train/')
        train_file = _resolve_file(train_data_dir, dataset, idx, phase)

        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].item()

        return train_data

    else:
        test_data_dir = os.path.join(_dataset_root(), dataset, 'test/')
        test_file = _resolve_file(test_data_dir, dataset, idx, phase)

        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].item()

        return test_data


def has_global_test(dataset):
    """True se o dataset possui conjunto de teste global (test/global_phase1.npz)."""
    return os.path.exists(os.path.join(_dataset_root(), dataset, 'test', 'global_phase1.npz'))


def read_global_test(dataset, phase="phase1"):
    """Lê o teste global (teste oficial completo) gravado pelo gerador ASTRA.

    phase1 = conceito original; phase2 = teste oficial com a transformação de drift aplicada.
    Retorna (X float32, y int64) como ``read_client_data``.
    """
    path = os.path.join(_dataset_root(), dataset, 'test', f"global_{phase}.npz")
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].item()
    X = torch.as_tensor(data['x'], dtype=torch.float32)
    y = torch.as_tensor(data['y'], dtype=torch.int64)
    del data
    return X, y


def read_client_data(dataset, idx, is_train=True, phase=None):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    data = read_data(dataset, idx, is_train, phase=phase)
    X = torch.as_tensor(data['x'], dtype=torch.float32)
    y = torch.as_tensor(data['y'], dtype=torch.int64)
    del data
    return X, y


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
