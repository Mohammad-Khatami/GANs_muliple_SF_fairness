import torch
import torch.nn as nn
import torch.optim as optim

def adv_loss_DP_vectorized(batch, actual, pred):
    # Calculate absolute differences
    abs_diff = torch.abs(actual - pred)

    # Masks for selecting groups
    mask_1 = actual > 0.5
    mask_0 = ~mask_1

    # Group counts
    D_1 = mask_1.sum().float()  # Ensure float for division
    D_0 = batch - D_1

    # Group losses
    loss_1 = abs_diff[mask_1].sum()
    loss_0 = abs_diff[mask_0].sum()

    # Compute adversary loss with safe division
    loss_1 = loss_1 / D_1 if D_1 > 0 else 0
    loss_0 = loss_0 / D_0 if D_0 > 0 else 0

    # Final adversarial loss calculation
    if D_1 == 0:
        adversary_loss_DP = 1.0 - loss_0
    elif D_0 == 0:
        adversary_loss_DP = 1.0 - loss_1
    else:
        adversary_loss_DP = 1.0 - loss_1 - loss_0

    return adversary_loss_DP

def adv_loss_EO_vectorized(adv_actual, adv_pred, label):
    # Convert binary classifications to float for computations
    adv_actual = adv_actual.float()
    adv_pred = adv_pred.float()
    label = label.float()

    # Create masks for each condition
    mask_1_1 = (adv_actual == 1) & (label == 1)
    mask_1_0 = (adv_actual == 1) & (label == 0)
    mask_0_1 = (adv_actual == 0) & (label == 1)
    mask_0_0 = (adv_actual == 0) & (label == 0)

    # Calculate losses in each category
    loss_1_1 = torch.abs(adv_pred - adv_actual)[mask_1_1].sum()
    loss_1_0 = torch.abs(adv_pred - adv_actual)[mask_1_0].sum()
    loss_0_1 = torch.abs(adv_pred - adv_actual)[mask_0_1].sum()
    loss_0_0 = torch.abs(adv_pred - adv_actual)[mask_0_0].sum()

    # Avoid division by zero
    D_1_1 = mask_1_1.sum().float().clamp(min=1)
    D_1_0 = mask_1_0.sum().float().clamp(min=1)
    D_0_1 = mask_0_1.sum().float().clamp(min=1)
    D_0_0 = mask_0_0.sum().float().clamp(min=1)

    # Combine losses for final EO adversarial loss
    EO_adv_loss = 2.0 - (
        loss_1_1 / D_1_1 + loss_1_0 / D_1_0 + loss_0_1 / D_0_1 + loss_0_0 / D_0_0
    )
    return EO_adv_loss

class CustomFarthestLoss_old(nn.Module):
    def __init__(self, distance_matrix):
        super(CustomFarthestLoss, self).__init__()
        self.distance_matrix = distance_matrix
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # Standard CE loss
    
    def forward(self, logits, targets):
        # Compute the standard cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Get predicted class (the class with the highest logit)
        _, predicted_class = torch.max(logits, dim=1)
        
        # Get the distances between the predicted class and the true class
        distances = self.distance_matrix[targets, predicted_class]
        
        # Modify the loss based on the distance: closer predictions have higher penalties
        modified_loss = ce_loss * distances
        
        return modified_loss.mean()  # Return the mean of the modified loss

class CustomCrossEntropyLoss_old(nn.Module):
    def __init__(self, weight_matrix):
        super(CustomCrossEntropyLoss, self).__init__()
        self.weight_matrix = weight_matrix
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # We’ll handle weight manually

    def forward(self, logits, targets):
        # Compute the normal CE loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Gather weights for the farthest class (based on targets)
        weights_for_targets = self.weight_matrix[targets]
        
        # Apply the weights dynamically
        weighted_loss = ce_loss * weights_for_targets.max(dim=1)[0]  # Max weight for farthest class
        
        return weighted_loss.mean()  # Return mean loss

class CustomFarthestLoss(nn.Module):
    def __init__(self):
        super(CustomFarthestLoss, self).__init__()
        # Define the distance matrix inside the class
        self.distance_matrix = torch.tensor([
            # NW-M   W-M    NW-F   W-F
            [1,     1,     1,     0.01],  # NW-M (0) -> W-F (3) is farthest (distance = 0)
            [1,     1,     0.01,     1],  # W-M (1) -> NW-F (2) is farthest (distance = 0)
            [1,     0.01,     1,     1],  # NW-F (2) -> W-M (1) is farthest (distance = 0)
            [0.01,     1,     1,     1],  # W-F (3) -> NW-M (0) is farthest (distance = 0)
        ], dtype=torch.float32)
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # Standard CrossEntropyLoss without reduction

    def forward(self, logits, targets):
        # Compute the standard cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Get predicted class (the class with the highest logit)
        _, predicted_class = torch.max(logits, dim=1)
        
        # Look up the distance between the predicted class and the true class
        distances = self.distance_matrix[targets, predicted_class]
        
        # Modify the loss based on the distance: closer predictions have higher penalties
        modified_loss = ce_loss * distances
        
        return modified_loss.mean()  # Return the mean of the modified loss


def compute_error_rates_binary(y_true, y_pred, sf):
    """ Helper function to calculate false positive and false negative rates """
    fpr_group1 = ((y_pred == 1) & (y_true == 0) & (sf == 1)).float().mean()
    fnr_group1 = ((y_pred == 0) & (y_true == 1) & (sf == 1)).float().mean()

    fpr_group2 = ((y_pred == 1) & (y_true == 0) & (sf == 0)).float().mean()
    fnr_group2 = ((y_pred == 0) & (y_true == 1) & (sf == 0)).float().mean()

    return fpr_group1, fnr_group1, fpr_group2, fnr_group2

def eo_loss_binary(fpr_group1, fnr_group1, fpr_group2, fnr_group2):
    """ Equalized Odds loss function: penalizes differences in error rates """
    fpr_diff = torch.abs(fpr_group1 - fpr_group2)
    fnr_diff = torch.abs(fnr_group1 - fnr_group2)
    return fpr_diff + fnr_diff



def compute_error_rates_multi_old(y_true, y_pred, sf):
    """ Helper function to calculate false positive and false negative rates for the 4 sensitive groups """
    # Assuming y_true, y_pred, and sf are all torch tensors and part of the computational graph

    y_pred_binary = (y_pred >= 0.5).float()  # Threshold at 0.5 to get binary predictions
    y_pred_binary = y_pred_binary.squeeze()
    y_true = y_true.squeeze()
    sf = sf.squeeze()

    # Initialize dictionaries to hold FPR and FNR for each group
    fpr = {}
    fnr = {}

    # Group: 0 -> NW-M (Non-White Male)
    fpr[0] = ((y_pred == 1) & (y_true == 0) & (sf == 0)).float().mean()
    fnr[0] = ((y_pred == 0) & (y_true == 1) & (sf == 0)).float().mean()

    # Group: 1 -> W-M (White Male)
    fpr[1] = ((y_pred == 1) & (y_true == 0) & (sf == 1)).float().mean()
    fnr[1] = ((y_pred == 0) & (y_true == 1) & (sf == 1)).float().mean()

    # Group: 2 -> NW-F (Non-White Female)
    fpr[2] = ((y_pred == 1) & (y_true == 0) & (sf == 2)).float().mean()
    fnr[2] = ((y_pred == 0) & (y_true == 1) & (sf == 2)).float().mean()

    # Group: 3 -> W-F (White Female)
    fpr[3] = ((y_pred == 1) & (y_true == 0) & (sf == 3)).float().mean()
    fnr[3] = ((y_pred == 0) & (y_true == 1) & (sf == 3)).float().mean()

    return fpr, fnr

def compute_error_rates_multi_old_v2(y_true, y_pred, sf):
    """ Helper function to calculate false positive and false negative rates for the 4 sensitive groups """
    # Assuming y_true, y_pred, and sf are all torch tensors and part of the computational graph

    # Apply sigmoid to y_pred to get probabilities (or if you have logits, use them accordingly)
    y_pred_prob = torch.sigmoid(y_pred).squeeze()  # Convert logits to probabilities
    y_true = y_true.squeeze()
    sf = sf.squeeze()

    # Initialize dictionaries to hold FPR and FNR for each group
    fpr = {}
    fnr = {}

    # Group: 0 -> NW-M (Non-White Male)
    fpr[0] = torch.masked_select(y_pred_prob, sf == 0).mean() * (1 - torch.masked_select(y_true, sf == 0)).mean()
    fnr[0] = ((1 - torch.masked_select(y_pred_prob, sf == 0)) * torch.masked_select(y_true, sf == 0)).mean()
   # fpr[0] = (y_pred_prob[sf == 0] * (1 - y_true[sf == 0])).mean()  # False positives for group 0
   # fnr[0] = ((1 - y_pred_prob[sf == 0]) * y_true[sf == 0]).mean()  # False negatives for group 0

    # Group: 1 -> W-M (White Male)
    fpr[1] = torch.masked_select(y_pred_prob, sf == 1).mean() * (1 - torch.masked_select(y_true, sf == 1)).mean()
    fnr[1] = ((1 - torch.masked_select(y_pred_prob, sf == 1)) * torch.masked_select(y_true, sf == 1)).mean()
    #fpr[1] = (y_pred_prob[sf == 1] * (1 - y_true[sf == 1])).mean()
    #fnr[1] = ((1 - y_pred_prob[sf == 1]) * y_true[sf == 1]).mean()

    # Group: 2 -> NW-F (Non-White Female)
    fpr[2] = torch.masked_select(y_pred_prob, sf == 2).mean() * (1 - torch.masked_select(y_true, sf == 2)).mean()
    fnr[2] = ((1 - torch.masked_select(y_pred_prob, sf == 2)) * torch.masked_select(y_true, sf == 2)).mean()
    #fpr[2] = (y_pred_prob[sf == 2] * (1 - y_true[sf == 2])).mean()
    #fnr[2] = ((1 - y_pred_prob[sf == 2]) * y_true[sf == 2]).mean()

    # Group: 3 -> W-F (White Female)
    fpr[3] = torch.masked_select(y_pred_prob, sf == 3).mean() * (1 - torch.masked_select(y_true, sf == 3)).mean()
    fnr[3] = ((1 - torch.masked_select(y_pred_prob, sf == 3)) * torch.masked_select(y_true, sf == 3)).mean()
    #fpr[3] = (y_pred_prob[sf == 3] * (1 - y_true[sf == 3])).mean()
    #fnr[3] = ((1 - y_pred_prob[sf == 3]) * y_true[sf == 3]).mean()
    return fpr, fnr

def compute_error_rates_multi(y_true, y_pred, sf):
    """
    Compute False Positive Rates (FPR) and False Negative Rates (FNR) for multiple sensitive groups.

    Args:
        y_true (torch.Tensor): Ground truth binary labels (shape: [batch_size]).
        y_pred (torch.Tensor): Predicted logits (shape: [batch_size]).
        sf (torch.Tensor): Sensitive feature labels (categorical, shape: [batch_size]).

    Returns:
        fpr (dict): False positive rates for each sensitive group.
        fnr (dict): False negative rates for each sensitive group.
    """
    # Convert logits to probabilities
    y_pred_prob = torch.sigmoid(y_pred).squeeze()
    y_true = y_true.squeeze()
    sf = sf.squeeze()

    # Initialize dictionaries for FPR and FNR
    fpr = {}
    fnr = {}

    # Iterate through sensitive groups (0, 1, 2, 3)
    for group in range(4):
        # Create mask for the current group
        mask = sf == group

        if mask.sum() > 0:  # Ensure there are samples for this group
            # Extract predictions and true labels for this group
            y_pred_group = torch.masked_select(y_pred_prob, mask)
            y_true_group = torch.masked_select(y_true, mask)

            # Compute FPR: False positives / Total negatives
            negatives = 1 - y_true_group
            false_positives = y_pred_group * negatives
            if negatives.sum() > 0:  # Check for zero negatives
                fpr[group] = false_positives.sum() / negatives.sum()
            else:
                fpr[group] = torch.tensor(0.0, device=y_pred_prob.device)

            # Compute FNR: False negatives / Total positives
            positives = y_true_group
            false_negatives = (1 - y_pred_group) * positives
            if positives.sum() > 0:  # Check for zero positives
                fnr[group] = false_negatives.sum() / positives.sum()
            else:
                fnr[group] = torch.tensor(0.0, device=y_pred_prob.device)
        else:
            # Assign default values if no samples are available
            fpr[group] = torch.tensor(0.0, device=y_pred_prob.device)
            fnr[group] = torch.tensor(0.0, device=y_pred_prob.device)

    return fpr, fnr



def eo_loss_multi(fpr, fnr):
    """ Equalized Odds loss function for the 4 sensitive groups (NW-M, W-M, NW-F, W-F) """
    fpr_diff_sum = torch.tensor(0.0, device=fpr[0].device)  # No need for requires_grad=True here
    fnr_diff_sum = torch.tensor(0.0, device=fnr[0].device)

    # List of class combinations for pairwise comparisons
    classes = [0, 1, 2, 3]

    # Calculate pairwise differences in FPR and FNR between each class
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            fpr_diff_sum = fpr_diff_sum + torch.abs(fpr[classes[i]] - fpr[classes[j]])
            fnr_diff_sum = fnr_diff_sum + torch.abs(fnr[classes[i]] - fnr[classes[j]])

    # Return the sum of differences in FPR and FNR across all groups
    return fpr_diff_sum + fnr_diff_sum


