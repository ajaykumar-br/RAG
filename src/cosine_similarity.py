import torch

def cosine_similarity(q, D):

    dot = D@q
    D_norm = torch.linalg.norm(D, dim=1)
    q_norm = torch.linalg.norm(q)

    return dot/(D_norm*q_norm)