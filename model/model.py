import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch



def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == "__main__":
    pass
