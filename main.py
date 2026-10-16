import torch
from src.cosine_similarity import cosine_similarity

def main():
    q = torch.tensor([1.0, 1.0, 0.0])
    D = torch.tensor([[2.0, 2.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0],
                    [-1.0, -1.0, 0.0]])
    print((D @ q).shape, torch.linalg.norm(D, dim=1).shape)
    print(cosine_similarity(q, D))

if __name__ == "__main__":
    main()
