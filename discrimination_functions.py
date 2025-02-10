import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def confusion_matrix_metrics(y_true, y_pred):
    # Ensure y_pred is binary (assumes y_pred is a probability score between 0 and 1)
#    y_pred = (y_pred >= 0.5).astype(int)
    y_pred = (y_pred >= 0.5).to(torch.int)


    # Handle the case where there is only one class in y_true or y_pred
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    if len(unique_true) == 1 or len(unique_pred) == 1:
        # If there is only one unique class in y_true or y_pred, the confusion matrix will be 1x1
        # We will return default values for the confusion matrix components
        if len(unique_true) == 1 and unique_true[0] == 0:  # Only 0s in y_true
            tp, tn, fp, fn = 0, len(y_true), 0, 0
        elif len(unique_true) == 1 and unique_true[0] == 1:  # Only 1s in y_true
            tp, tn, fp, fn = len(y_true), 0, 0, 0
        else:
            # If y_pred has only one class, handle similarly
            tp, tn, fp, fn = 0, len(y_true), 0, 0
    else:
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tp = cm[1, 1]  # True Positives
        tn = cm[0, 0]  # True Negatives
        fp = cm[0, 1]  # False Positives
        fn = cm[1, 0]  # False Negatives

    # Create a dictionary to store metrics
    metrics = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }

    return metrics

def single_disc_DP_per_batch_multi(predicted_label, sf,gen:bool=True):

    true_sf_1 = true_sf_0 = 0
    _Y_hat_1_sf_1 = _Y_hat_1_sf_0 =0 
    # ---------
    _salary_predicted_to_be_above_50k = (predicted_label >= 0.5).squeeze()
    _Y_hat_1 = (_salary_predicted_to_be_above_50k)  # > 0.5

    if gen:
        _sf_1 = (sf > 1.5)
    else:
        _sf_1 = (sf == 1)|(sf == 3)

    _sf_0 = (~_sf_1)
    # --------- 
    true_sf_1 = (_sf_1).sum()  
    true_sf_0 = (_sf_0).sum()  
    _Y_hat_1_sf_1 = ((_Y_hat_1) & (_sf_1)).sum()  
    _Y_hat_1_sf_0 = ((_Y_hat_1) & (_sf_0)).sum()  

    return [
        true_sf_1,
        true_sf_0,
        _Y_hat_1_sf_1,
        _Y_hat_1_sf_0,
    ]
def double_disc_DP_per_batch(predicted_label,  sf1, sf2):

    true_sf1_1_sf2_1 = true_sf1_1_sf2_0 = true_sf1_0_sf2_1 = true_sf1_0_sf2_0 = 0 
    _Y_hat_1_sf1_1_sf2_1 = _Y_hat_1_sf1_1_sf2_0 = _Y_hat_1_sf1_0_sf2_1 = _Y_hat_1_sf1_0_sf2_0 = 0
    # --------- 
    _Y_hat_1 = (predicted_label >= 0.5).squeeze()
    _sf1_1 = (sf1 >= 0.5)
    _sf1_0 = (~_sf1_1)
    _sf2_1 = (sf2 >= 0.5)
    _sf2_0 = (~_sf2_1)
    # --------- find true race/gender conditioned on true gender/race
    true_sf1_1_sf2_1 = ((_sf1_1) & (_sf2_1)).sum()  # gender=1,race=1
    true_sf1_1_sf2_0 = ((_sf1_1) & (_sf2_0)).sum()  # gender=1,race=0
    true_sf1_0_sf2_1 = ((_sf1_0) & (_sf2_1)).sum()  # gender=0,race=1
    true_sf1_0_sf2_0 = ((_sf1_0) & (_sf2_0)).sum()  # gender=0,race=0
    # --------- 
    _Y_hat_1_sf1_1_sf2_1 = ((_Y_hat_1) & (_sf1_1) & (_sf2_1)).sum()  # Y_hat_1, gender=1,race=1
    _Y_hat_1_sf1_1_sf2_0 = ((_Y_hat_1) & (_sf1_1) & (_sf2_0)).sum()  # Y_hat_1, gender=1,race=0
    _Y_hat_1_sf1_0_sf2_1 = ((_Y_hat_1) & (_sf1_0) & (_sf2_1)).sum()  # Y_hat_1, gender=0,race=1
    _Y_hat_1_sf1_0_sf2_0 = ((_Y_hat_1) & (_sf1_0) & (_sf2_0)).sum()  # Y_hat_1, gender=0,race=0

    return [
        true_sf1_1_sf2_1,
        true_sf1_1_sf2_0,
        true_sf1_0_sf2_1,
        true_sf1_0_sf2_0,
        _Y_hat_1_sf1_1_sf2_1,
        _Y_hat_1_sf1_1_sf2_0,
        _Y_hat_1_sf1_0_sf2_1,
        _Y_hat_1_sf1_0_sf2_0,
    ]

def single_disc_EO_per_batch_multi(predicted_label, label, sf, gen:bool=True):

    _sf_1_Y_0 = _sf_1_Y_1 = _sf_0_Y_0 = _sf_0_Y_1 = 0
    _Y_hat_1_sf_1_Y_0 = _Y_hat_1_sf_1_Y_1 = _Y_hat_1_sf_0_Y_0 = _Y_hat_1_sf_0_Y_1 = 0
    # ---------
    _Y_hat_1 = (predicted_label >= 0.5).squeeze()
    _Y_1 = (label >= 0.5)
    _Y_0 = (~_Y_1)
    if gen:
        _sf_1 = (sf > 1.5)
    else:
        _sf_1 = (sf == 1)|(sf == 3)

    _sf_0 = (~_sf_1)
    # ---------
    _sf_1_Y_0 += ((_sf_1) & (_Y_0)).sum()  
    _sf_1_Y_1 += ((_sf_1) & (_Y_1)).sum()  
    _sf_0_Y_0 += ((_sf_0) & (_Y_0)).sum()  
    _sf_0_Y_1 += ((_sf_0) & (_Y_1)).sum()  
    # ---------
    _Y_hat_1_sf_1_Y_0 +=((_Y_hat_1) & (_sf_1) & (_Y_0)).sum()  # (Y_h=1|SF=1,Y=0)
    _Y_hat_1_sf_1_Y_1 +=((_Y_hat_1) & (_sf_1) & (_Y_1)).sum()  # (Y_h=1|SF=1,Y=1)
    _Y_hat_1_sf_0_Y_0 +=((_Y_hat_1) & (_sf_0) & (_Y_0)).sum()  # (Y_h=1|SF=0,Y=0)
    _Y_hat_1_sf_0_Y_1 +=((_Y_hat_1) & (_sf_0) & (_Y_1)).sum()  # (Y_h=1|SF=0,Y=1)
  #  print(_Y_hat_1,_Y_1, _sf_1)
    return [
        _sf_1_Y_0,
        _sf_1_Y_1,
        _sf_0_Y_0,
        _sf_0_Y_1,
        _Y_hat_1_sf_1_Y_0,
        _Y_hat_1_sf_1_Y_1,
        _Y_hat_1_sf_0_Y_0,
        _Y_hat_1_sf_0_Y_1,
    ]

def double_disc_EO_per_batch_multi(predicted_label, label, sf):

    predicted_label=predicted_label.squeeze()

    _sf1_1_Y0_sf2_0 =_sf1_1_Y1_sf2_0 =_sf1_0_Y0_sf2_0 =_sf1_0_Y1_sf2_0 = 0
    pred_Y1_sf2_1_Y0_sf2_0 = pred_Y1_sf2_1_Y1_sf2_0 = pred_Y1_sf2_0_Y0_sf2_0 = (
        pred_Y1_sf2_0_Y1_sf2_0
    ) = 0
    _sf1_1_Y0_sf2_1 =_sf1_1_Y1_sf2_1 =_sf1_0_Y0_sf2_1 =_sf1_0_Y1_sf2_1 = 0
    pred_Y1_sf2_1_Y0_sf2_1 = pred_Y1_sf2_1_Y1_sf2_1 = pred_Y1_sf2_0_Y0_sf2_1 = (
        pred_Y1_sf2_0_Y1_sf2_1
    ) = 0

    _Y_1 = (label >= 0.5)
    _Y_0 = (~_Y_1)
    _Y_hat_1 = (predicted_label >= 0.5)

    _sf1_1 = (sf > 1.5)
    _sf1_0 = (~_sf1_1)

    _sf2_1 = (sf == 1)|(sf == 3)
    _sf2_0 = (~_sf2_1)
   
    # --------- find true gender & Y |race=0
    _sf1_1_Y0_sf2_0 += ((_sf1_1) & (_Y_0) & (_sf2_0)).sum()
    _sf1_1_Y1_sf2_0 += ((_sf1_1) & (_Y_1) & (_sf2_0)).sum()
    _sf1_0_Y0_sf2_0 += ((_sf1_0) & (_Y_0) & (_sf2_0)).sum()
    _sf1_0_Y1_sf2_0 += ((_sf1_0) & (_Y_1) & (_sf2_0)).sum()
    # -------- find predicted sensitive feature gender  & Y |race=0
    pred_Y1_sf2_1_Y0_sf2_0 += ((_Y_hat_1) & (_sf1_1) & (_Y_0) & (_sf2_0)).sum()
    pred_Y1_sf2_0_Y0_sf2_0 += ((_Y_hat_1) & (_sf1_0) & (_Y_0) & (_sf2_0)).sum()
    pred_Y1_sf2_1_Y1_sf2_0 += ((_Y_hat_1) & (_sf1_1) & (_Y_1) & (_sf2_0)).sum()
    pred_Y1_sf2_0_Y1_sf2_0 += ((_Y_hat_1) & (_sf1_0) & (_Y_1) & (_sf2_0)).sum()
    # --------- find true gender & Y |race=1
    _sf1_1_Y0_sf2_1 += ((_sf1_1) & (_Y_0) & (_sf2_1)).sum()
    _sf1_1_Y1_sf2_1 += ((_sf1_1) & (_Y_1) & (_sf2_1)).sum()
    _sf1_0_Y0_sf2_1 += ((_sf1_0) & (_Y_0) & (_sf2_1)).sum()
    _sf1_0_Y1_sf2_1 += ((_sf1_0) & (_Y_1) & (_sf2_1)).sum()
    # -------- find predicted sensitive feature gender  & Y |race=1
    pred_Y1_sf2_1_Y0_sf2_1 += ((_Y_hat_1) & (_sf1_1) & (_Y_0) & (_sf2_1)).sum()
    pred_Y1_sf2_0_Y0_sf2_1 += ((_Y_hat_1) & (_sf1_0) & (_Y_0) & (_sf2_1)).sum()
    pred_Y1_sf2_1_Y1_sf2_1 += ((_Y_hat_1) & (_sf1_1) & (_Y_1) & (_sf2_1)).sum()
    pred_Y1_sf2_0_Y1_sf2_1 += ((_Y_hat_1) & (_sf1_0) & (_Y_1) & (_sf2_1)).sum()
    # --------------------------------------------------F4

    return [
       _sf1_1_Y0_sf2_0,
       _sf1_1_Y1_sf2_0,
       _sf1_0_Y0_sf2_0,
       _sf1_0_Y1_sf2_0,
       _sf1_1_Y0_sf2_1,
       _sf1_1_Y1_sf2_1,
       _sf1_0_Y0_sf2_1,
       _sf1_0_Y1_sf2_1,
        pred_Y1_sf2_1_Y0_sf2_0,
        pred_Y1_sf2_1_Y1_sf2_0,
        pred_Y1_sf2_0_Y0_sf2_0,
        pred_Y1_sf2_0_Y1_sf2_0,
        pred_Y1_sf2_1_Y0_sf2_1,
        pred_Y1_sf2_1_Y1_sf2_1,
        pred_Y1_sf2_0_Y0_sf2_1,
        pred_Y1_sf2_0_Y1_sf2_1,
    ]



