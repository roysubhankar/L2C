import numpy as np

import torch
import torch.nn as nn
from modules.distance import KLDiv


class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(KCL,self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        kld = self.kld(prob1,prob2)
        output = self.hingeloss(kld,simi)
        return output


class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
        
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))

        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(MCL.eps).log_()
        return neglogP.mean()

class DPS(nn.Module):
    # dense pair similarity loss
    def __init__(self):
        super(DPS, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # avoid dividing by zero
        self.eps = 1e-7

    def forward(self, x, target):
        # mask of 1s for the positives
        similar_pair_mask = (target == 1.).float()
        # mask of 1s for the negatives
        dissimilar_pair_mask = (target == 0.).float()

        # subsample the dissimilar_pair_mask
        subsampled_idxs = np.random.choice(np.where(dissimilar_pair_mask.cpu().numpy() == 1.)[0], 
                                           size=int(similar_pair_mask.sum().item() * 2), replace=True)
        
        # initialize an mask of all zeros. This mask will indicate which samples 
        # to BP
        final_mask = torch.zeros_like(similar_pair_mask)
        # fill the sub-sampled indices with 1s
        final_mask[subsampled_idxs] = 1.

        # combine with the mask of positives
        final_mask += similar_pair_mask

        out = (self.criterion(x, target) * final_mask).sum() / (final_mask.sum() + self.eps)

        return out