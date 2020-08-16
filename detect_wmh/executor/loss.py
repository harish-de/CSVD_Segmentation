import torch

class loss():
    smooth = 1.

    def dice_coef_for_training(self, y_pred, y_true):
        '''
        :param y_pred: predicted output
        :param y_true: actual ground truth
        :return: Dice Similarity coefficient score
        '''
        y_true_f = y_true.contiguous().view(-1)
        y_pred_f = y_pred.contiguous().view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def dice_coef_loss(self, y_pred, y_true):
        '''
        :return: return dice loss score
        '''
        return 1. - self.dice_coef_for_training(y_pred, y_true)