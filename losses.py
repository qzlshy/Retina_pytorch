import torch
import torch.nn as nn

class Focalloss(nn.Module):
    def __init__(self,alpha=0.25, gamma=2.0):
        super(Focalloss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.BCE=torch.nn.BCELoss(reduction='none')

    def forward(self, y_pred, y_true):
        labels = y_true[:,:-1]
        anchor_state = y_true[:,-1]
        classification = y_pred

        cs=anchor_state!=-1
        labels=labels[cs]
        classification=classification[cs]

        focal_weight = labels*(self.alpha)*((1.0-classification)**self.gamma)+(1.0-labels)*(1.0-self.alpha)*(classification**self.gamma)
        bce=self.BCE(classification,labels)
        loss = focal_weight * bce
        loss_all=torch.sum(loss)
        normalizer=torch.sum(anchor_state==1)
        normalizer=torch.clamp(normalizer, min=1.0)
        return loss_all/normalizer


class Smooth_l1_loss(nn.Module):
    def __init__(self,sigma=3.0):
        super(Smooth_l1_loss, self).__init__()
        self.sigma_squared = sigma**2

    def forward(self,y_pred, y_true):
        regression        = y_pred
        regression_target = y_true[:, :-1]
        anchor_state      = y_true[:, -1]
        cs=anchor_state==1
        regression=regression[cs]
        regression_target=regression_target[cs]
        regression_diff = regression - regression_target
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 /  self.sigma_squared),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 /  self.sigma_squared
                )
        loss_all=torch.sum(regression_loss)
        normalizer = torch.sum(cs)
        normalizer=torch.clamp(normalizer, min=1.0)
        return loss_all/normalizer
