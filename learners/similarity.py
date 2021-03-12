from types import MethodType
import torch
import torch.nn as nn
import models
from .classification import Learner_Classification as Learner_Template
from modules.pairwise import PairEnum


class Learner_DensePairSimilarity(Learner_Template):

    @staticmethod
    def create_model(model_type,model_name,out_dim):
        # Create Similarity Prediction Network (SPN) by model surgery
        model = models.__dict__[model_type].__dict__[model_name](out_dim=out_dim)
        n_feat = model.last.in_features

        # Replace task-dependent module
        model.last = nn.Sequential(
            nn.Linear(n_feat*2, n_feat*4),
            nn.BatchNorm1d(n_feat*4),
            nn.ReLU(inplace=True),
            nn.Linear(n_feat*4, 2)
        )

        # Replace task-dependent function
        def new_logits(self, x):
            feat1, feat2 = PairEnum(x)
            featcat = torch.cat([feat1, feat2], 1)
            out = self.last(featcat)
            return out
        model.logits = MethodType(new_logits, model)

        return model

class Learner_EdgeNetwork(Learner_Template):

    @staticmethod
    def create_model(model_type,model_name,out_dim):
        # Create Edge Network (EN) by model surgery
        model = models.__dict__[model_type].__dict__[model_name](out_dim=out_dim)
        n_feat = model.last.in_features

        # Replace task-dependent module
        model.last = nn.Sequential(
            nn.Conv2d(n_feat, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 2, kernel_size=(1, 1), stride=(1, 1))
        )

        # Replace task-dependent function
        def new_logits(self, x):
            x = x.unsqueeze(dim=0)
            x_i = x.unsqueeze(2)
            x_j = torch.transpose(x_i, 1, 2)
            x_ij = torch.abs(x_i - x_j)
            x_ij = torch.transpose(x_ij, 1, 3)
            sim_val = self.last(x_ij).squeeze(0)
            out = sim_val.permute(1, 2, 0).view(-1, 2)
            return out
        model.logits = MethodType(new_logits, model)

        return model