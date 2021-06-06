import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch


class BaselineModel(BaseModel):
    def __init__(self,
                 experts_used,
                 expert_dims,
                 projection_dim,
                 text_params):
        super().__init__()

        self.experts_used = experts_used
        self.video_GU = nn.ModuleDict({
            expert: Gated_Embedding_Unit(expert_dims[expert], projection_dim)
            for expert in experts_used
        })

        txt_dim = text_params['dim']
        self.text_GU = nn.ModuleDict({
            expert: Gated_Embedding_Unit(txt_dim, projection_dim, channels=0)
            for expert in experts_used

        })


    def forward(self, x):
        for expert in self.experts_used:
            ftr = x[expert]['ftr']


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True, channels=0):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension, channels)
        self.gating = gating

    def forward(self, x):
        x = self.fc(x)
        if self.gating:
            x = self.cg(x)
        x = F.normalize(x, dim=-1)

        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension, channels, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.channels = channels
        if channels > 0:
            bn_dim = channels
        else:
            bn_dim = dimension
        self.batch_norm = nn.BatchNorm1d(bn_dim)

    def forward(self, x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1)

        x = torch.cat((x, x1), -1)

        return F.glu(x, -1)


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
