from csv import DictWriter
import os.path
import torch.nn as nn
import torch
from copy import deepcopy
from AAE_model import AdversarialAutoEncoder, params_optims
from utils import (generate_filename,mv_file,accuracy_calculator_per_batch,set_seed)
from loss_functions import (adv_loss_DP_vectorized, adv_loss_EO_vectorized, CustomFarthestLoss,compute_error_rates_multi, eo_loss_multi)
from discrimination_functions import (single_disc_DP_per_batch_multi,single_disc_EO_per_batch_multi,
                                      double_disc_EO_per_batch_multi,confusion_matrix_metrics)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(
    #ML_model: AdversarialAutoEncoder,
    train_loader,
    val_loader,
    batch,
    alpha,
    beta,
    gamma,
    stack_number,
    config: dict,
    metric_type="DP",# Metric type can be "DP" or "EO"
    epoch_num=10,
    termination_epoch_threshold: int | None = None,
    margin_threshold: float | None = None,
    lr=0.001,
    log_filename_tail: str = "",
    #**kwargs,
):
    if not termination_epoch_threshold:
        termination_epoch_threshold = 10000 # float("inf")
    if not margin_threshold:
        margin_threshold = 0.5

    results=[]
    random_seeds=[1, 7, 42, 65, 92, 110, 313]
    for seed in random_seeds:
        set_seed(seed)
        if stack_number==1:
            ML_model = AdversarialAutoEncoder(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim_level_1"],
                encoded_dim=config["encoded_dim_level_1"],
                num_layers=config["num_layers"],
                dropout_rate=config["dropout_rate"],
                stack_number=stack_number,
            )
            print(f"stack number:{stack_number}, ML_model.stack_number: {ML_model.stack_number}")
        elif stack_number==2:
            ML_model = AdversarialAutoEncoder(
                input_dim=config["encoded_dim_level_1"],
                hidden_dim=config["hidden_dim_level_2"],
                encoded_dim=config["encoded_dim_level_2"],
                num_layers=config["num_layers"],
                dropout_rate=config["dropout_rate"],
                stack_number=stack_number,
            )
            print(f"stack number:{stack_number}, ML_model.stack_number: {ML_model.stack_number}")
        

        ML_model.to(device=device)


        _highest_val_acc = -1
        _epochs_passed_without_improvement = 0
        _lowest_delta = float("inf")
        _chosen_model = deepcopy(ML_model)

        log_filename = generate_filename(
            config={
                "alpha": alpha,
                "beta": beta,
                "num_layers": config["num_layers"],
                "gamma": gamma,
                "epoch_num": epoch_num,
                "seed": seed,
                "lr": config["lr"],
                "dropout_rate": config["dropout_rate"],
                "batch_size": config["batch_size"],
            },
            stack=ML_model.stack_number,
            output_source="train",
            discrimination_metric=metric_type,  # Correctly use the dynamic metric_type variable
            trailing_text=log_filename_tail,
            file_extension=".csv"
        )

        log_file = open(log_filename, "w")
        writer = DictWriter(
            log_file,
            fieldnames=[
                "epoch",
                "avg_loss",
                "val_loss",
                "avg_accuracy",
                "val_acc",
                "y_disc_gen",
                "y_disc_race",
            ],
            dialect="unix",
        )
        writer.writeheader()
        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()
        criterion_adv = CustomFarthestLoss()
        optimizer_encoder, optimizer_decoder, optimizer_predictor, optimizer_adv_multi = params_optims(ML_model, lr)

        # Training variables
        train_loss_all, valid_loss_all = [], []
        train_acc_all, valid_acc_all = [], []
        valid_disc_gender_all, valid_disc_race_all = [], []


        # Training loop
        for epoch in range(epoch_num):
            ML_model.train()
            train_loss, total_accuracy, U = 0.0, 0, 1
            train_error=0.0
            for j, (X, Y, gender_race) in enumerate(train_loader, 0):
                X_hat, Y_hat, gender_race_hat = ML_model(X)
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                optimizer_predictor.zero_grad()
                optimizer_adv_multi.zero_grad()
 
                # Calculate losses
                loss_bce = criterion_bce(Y_hat.squeeze(), Y)
                loss_mse = criterion_mse(X_hat, X)
                loss_adv_gender_race = criterion_adv(gender_race_hat, gender_race.long())

                loss_adv = beta * loss_adv_gender_race
                loss_encoder_decoder_predictor = alpha * loss_mse + gamma * loss_bce
                total_loss = loss_encoder_decoder_predictor + loss_adv
 
                train_loss += total_loss.item()
 
                #print(f"epoch: {epoch}, j: {j}, U: {U}, total_loss:{total_loss},loss_adv:{loss_adv}, " )
                if metric_type == "EO":
                    fpr, fnr = compute_error_rates_multi(Y, Y_hat, gender_race)
                    #eo_loss_value = (beta/2) * (1-eo_loss_multi(fpr, fnr))
                    loss_adv= loss_adv #+eo_loss_value
                    total_loss= loss_encoder_decoder_predictor + loss_adv
                error=gamma * loss_bce + loss_adv
                train_error+= error.item() 
                # Backpropagation
                #print(f"epoch: {epoch}, j: {j}, U: {U}, total_loss:{total_loss},loss_adv:{loss_adv}, eo_loss_value:{eo_loss_value},fpr: {fpr}, fnr:{fnr}   " )
                if U:
                    total_loss.backward()
                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    optimizer_predictor.step()
                else:
                    loss_adv.backward()
                    optimizer_adv_multi.step()
                U = not U
 
                # Calculate accuracy
                predicted = (Y_hat > 0.5).float()
                total_accuracy += (predicted.squeeze() == Y).float().mean().item()
 
            avg_loss = train_loss / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)
            avg_error_train=train_error/len(train_loader)

            valid_filename = generate_filename(
                config={
                    "alpha": alpha,
                    "beta": beta,
                    "num_layers": config["num_layers"],
                    "gamma": gamma,
                    "epoch_num": epoch_num,
                    "seed": seed,
                    "lr": lr,
                    "dropout_rate": config["dropout_rate"],
                    "batch_size": config["batch_size"],
                },
                stack=ML_model.stack_number,
                output_source="validation",
                discrimination_metric=metric_type,  # Correctly use the dynamic metric_type variable
                trailing_text=log_filename_tail,
                file_extension=".csv",
            )
 
            if metric_type == "DP":
                val_loss, val_acc, y_disc_gen, y_disc_race, val_error = test_model_DP(
                    ML_model, val_loader, criterion_bce, criterion_mse, batch, alpha, beta, gamma,
                    log_filename=valid_filename, epoch=epoch
                )
            else:
                val_loss, val_acc, y_disc_gen, y_disc_race, val_error = test_model_EO(
                    ML_model, val_loader, criterion_bce, criterion_mse, batch, alpha, beta, gamma,
                    log_filename=valid_filename, epoch=epoch
                )
 
 
            # Logging metrics
            train_loss_all.append(avg_loss)
            valid_loss_all.append(val_loss)
            train_acc_all.append(avg_accuracy)
            valid_acc_all.append(val_acc)
            valid_disc_gender_all.append(y_disc_gen)
            valid_disc_race_all.append(y_disc_race)
 
            writer.writerow(dict(
                epoch=epoch, avg_loss=avg_loss, val_loss=val_loss,
                avg_accuracy=avg_accuracy, val_acc=val_acc,
                y_disc_gen=y_disc_gen.item(), y_disc_race=y_disc_race.item()
            ))
 
            if abs(val_loss) < 1e-6:
                margin=0.0
            elif (val_loss - avg_loss) >0 :
                margin = abs((val_loss - avg_loss) / val_loss)
            else:
                margin = -1.0*abs((val_loss - avg_loss) / val_loss)
 
 
#### ## check and see if system is not nornmal!
 
            if margin > margin_threshold:
                _epochs_passed_without_improvement += 1
            else:
                _epochs_passed_without_improvement = 0
 
            print(
                f"Epoch {epoch+1}/{epoch_num}, Train Loss: {avg_loss:.6f}, Valid Loss: {val_loss:.6f}, Margin: {margin:.3f}, counter: {_epochs_passed_without_improvement}, train_acc:{avg_accuracy}, valid_acc:{val_acc}"
                 )
 
#### ## if system continues to be chaotic!
            if _epochs_passed_without_improvement == 0:
                _chosen_model = deepcopy(ML_model)
            
            if epoch %50==0:
                #avg_error_train
                #val_error
                results.append({
                    'seed': seed,
                    'epoch': epoch,
                    'model': _chosen_model,
                    'loss': val_loss, #avg_loss
                })

            if _epochs_passed_without_improvement == termination_epoch_threshold:
                print(f"\tBreaking after {_epochs_passed_without_improvement} epochs")
                ML_model.load_state_dict(
                _chosen_model.state_dict()
                )  # get the weights from the best model
                break


        log_file.close()
        _chars_to_trim = len(log_filename_tail) + 5
        mv_file(log_filename, log_filename[:-_chars_to_trim] + ".csv")
        mv_file(valid_filename, valid_filename[:-_chars_to_trim] + ".csv")

    return train_loss_all, valid_loss_all, train_acc_all, valid_acc_all, valid_disc_gender_all, valid_disc_race_all, results # _chosen_model

# Define the function to select the best model
def select_best_model(item_result):
    """
    Selects the best model based on the "median-like" loss for each epoch group.
    - Handles odd/even cases when calculating the median.
    - Finds the model with the lowest loss among all epoch group medians.
    Returns only the selected model.
    """
    # Group results by epoch
    epoch_groups = {}
    for result in item_result:
        epoch = result['epoch']
        if epoch not in epoch_groups:
            epoch_groups[epoch] = []
        epoch_groups[epoch].append(result)

    median_models = []

    # Function to calculate "median-like" value
    def calculate_median_like(results):
        """
        For a list of results:
        - If odd, return the median result.
        - If even, return the result with the lowest loss among the two middle elements.
        """
        sorted_results = sorted(results, key=lambda x: x['loss'])
        n = len(sorted_results)
        if n % 2 == 1:  # Odd number of results
            return sorted_results[n // 2]
        else:  # Even number of results
            mid1, mid2 = sorted_results[n // 2 - 1], sorted_results[n // 2]
            return mid1 if mid1['loss'] < mid2['loss'] else mid2

    # Calculate median-like model for each epoch group
    for epoch, results in epoch_groups.items():
        if results:  # Ensure there are results for this epoch group
            median_model = calculate_median_like(results)
            median_models.append(median_model)

    # Find the best model among the median models
    best_model_info = min(median_models, key=lambda x: x['loss'])

    # Print details about the best model
    #print(f"Best model: {best_model_info}")
    #print("---###---", best_model_info['model'])

    # Return the actual model (weights, NN structure, etc.)
    return best_model_info['model'], best_model_info['epoch'], best_model_info['seed']


def test_model_DP(
    ML_model: AdversarialAutoEncoder,
    loader,
    criterion_bce,
    criterion_mse,
    batch,
    alpha,
    beta,
    gamma,
    log_filename,
    epoch =1,
):
    _write_header = True
    # avoid writing header multiple times
    if os.path.exists(log_filename):
        _write_header = False

    log_file = open(log_filename, "a")
    writer = DictWriter(
        log_file,
        fieldnames=[
            "epoch",
            "avg_loss",
            "avg_accuracy",
            "y_disc_gender",
            "y_disc_race",
            "y_disc_gender_white",
            "y_disc_gender_non_white",
            "y_disc_female_race",
            "y_disc_male_race",
            "y_disc_avg_race_gender",
            "y_disc_abs_race_gender",
            "Y_TP", 
            "Y_TN", 
            "Y_FP", 
            "Y_FN", 
            "loss_bce",
            "loss_mse",
            "loss_adv_gender_race",
            "total_loss",
        ],
        dialect="unix",
    )
    if _write_header:
        writer.writeheader()

    ML_model.eval()
    total_accuracy = 0.0
    total_loss = 0.0

    # --------------------------
    Y_hat_1_female = 0
    Y_hat_1_male = 0
    true_female = 0
    true_male = 0
    Y_hat_1_white = 0
    Y_hat_1_non_white = 0
    true_white = 0
    true_non_white = 0
    # correct = 0
    # total = 0
    true_female_white = 0
    true_female_non_white = 0
    true_male_white = 0
    true_male_non_white = 0
    Y_hat_1_female_white = 0
    Y_hat_1_female_non_white = 0
    Y_hat_1_male_white = 0
    Y_hat_1_male_non_white = 0
    TP = TN = FP = FN = 0
    # --------------------------
    val_error=0.0
    criterion_adv = CustomFarthestLoss() 
    with torch.inference_mode():
        for i, (X, Y, gender_race) in enumerate(loader):
            X_hat, Y_hat, gender_race_hat = ML_model(X)

            gender_race=gender_race.long()
            loss_bce = criterion_bce(Y_hat.squeeze(), Y)
            loss_mse = criterion_mse(X_hat, X)
            loss_adv_gender_race = criterion_adv(gender_race_hat,gender_race)
             
            loss_adv= beta * loss_adv_gender_race
            loss_encoder_decoder_predictor=alpha * loss_mse + gamma * loss_bce

            loss = (
                  loss_encoder_decoder_predictor
                + loss_adv
                   )
            val_error+= (beta * loss_adv_gender_race +gamma * loss_bce).item()
            total_loss += loss.item()
            # Calculate accuracy
            predicted = (Y_hat > 0.5).float()  # Threshold predictions
            total_accuracy += (predicted.squeeze() == Y).float().mean().item()
            # ===================================================================
            # cache variables for performance
            _salary_predicted_to_be_above_50k = (Y_hat >= 0.5)
            _gender_is_female = (gender_race > 1.5)
            _gender_is_male = (~_gender_is_female)
            _race_is_white = (gender_race == 1)|( gender_race == 3)
            _race_is_not_white = (~_race_is_white)
            _Y_hat_binary_gt_0_5 = (_salary_predicted_to_be_above_50k)  # > 0.5
            _Y_hat_binary_le_0_5 = (~_Y_hat_binary_gt_0_5)

#            total += Y.size(0)  # calculates the total length of the dataset items
#            results = (
#                _salary_predicted_to_be_above_50k.squeeze() == Y
#            )  ## to find accuracy

#            correct += results.sum()  ## to find accuracy


            # --------- gender
            T_gen = single_disc_DP_per_batch_multi(predicted_label=Y_hat, sf=gender_race, gen=True)
            true_female += T_gen[0]
            true_male += T_gen[1]
            Y_hat_1_female += T_gen[2]
            Y_hat_1_male += T_gen[3]

            # --------- race
            T_race = single_disc_DP_per_batch_multi(predicted_label=Y_hat, sf=gender_race, gen=False)
            true_white += T_race[0]
            true_non_white += T_race[1]
            Y_hat_1_white += T_race[2]
            Y_hat_1_non_white += T_race[3]

            # --------- find true race/gender conditioned on true gender/race
            #T_gen_race= double_disc_DP_per_batch(predicted_label=Y_hat, sf1=gender,sf2=race)
            #true_female_white += T_gen_race[0]
            #true_female_non_white += T_gen_race[1]
            #true_male_white += T_gen_race[2]
            #true_male_non_white += T_gen_race[3]
            #Y_hat_1_female_white += T_gen_race[4]
            #Y_hat_1_female_non_white += T_gen_race[5]
            #Y_hat_1_male_white += T_gen_race[6]
            #Y_hat_1_male_non_white += T_gen_race[7]
            true_female_white += (gender_race==3).sum()
            true_female_non_white += (gender_race==2).sum()
            true_male_white += (gender_race==1).sum()
            true_male_non_white += (gender_race==0).sum()

            Y_hat_1_female_white += ((Y_hat.squeeze() >= 0.5)&(gender_race==3)).sum()                                    
            Y_hat_1_female_non_white += ((Y_hat.squeeze() >= 0.5)&(gender_race==2)).sum()     
            Y_hat_1_male_white += ((Y_hat.squeeze() >= 0.5)&(gender_race==1)).sum()                                    
            Y_hat_1_male_non_white += ((Y_hat.squeeze() >= 0.5)&(gender_race==0)).sum()                                    
            # ---------
            confusion = confusion_matrix_metrics(y_true=Y,y_pred=Y_hat)#,class_names=['0', '1'])
            TP += confusion.get("TP")
            TN += confusion.get("TN")
            FP += confusion.get("FP")
            FN += confusion.get("FN")

    y_disc_gender = abs(Y_hat_1_male / true_male - Y_hat_1_female / true_female)
    y_disc_race = abs(Y_hat_1_non_white / true_non_white - Y_hat_1_white / true_white)

    # # finding y_disc for double sensitive features
    y_disc_gender_white = abs(Y_hat_1_male_white / true_male_white - Y_hat_1_female_white / true_female_white)
    y_disc_gender_non_white = abs(Y_hat_1_male_non_white / true_male_non_white - Y_hat_1_female_non_white / true_female_non_white)
    y_disc_female_race = abs(Y_hat_1_female_non_white / true_female_non_white - Y_hat_1_female_white / true_female_white)
    y_disc_male_race = abs(Y_hat_1_male_non_white / true_male_non_white - Y_hat_1_male_white / true_male_white)

    y_disc_avg_race_gender = (
        y_disc_gender_white
        + y_disc_gender_non_white
        + y_disc_female_race
        + y_disc_male_race) / 4
    y_disc_abs_race_gender = (
        y_disc_gender_white
        + y_disc_gender_non_white
        + y_disc_female_race
        + y_disc_male_race)

    # ===================================================================
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    avg_val_error =val_error/len(loader)

    writer.writerow(
        dict(
            epoch= epoch,
            avg_loss=avg_loss,
            avg_accuracy=avg_accuracy,
            y_disc_gender=y_disc_gender.item(),
            y_disc_race=y_disc_race.item(),
            y_disc_gender_white=y_disc_gender_white.item(),
            y_disc_gender_non_white=y_disc_gender_non_white.item(),
            y_disc_female_race=y_disc_female_race.item(),
            y_disc_male_race=y_disc_male_race.item(),
            y_disc_avg_race_gender=y_disc_avg_race_gender.item(),
            y_disc_abs_race_gender=y_disc_abs_race_gender.item(),
            Y_TP=TP,
            Y_TN=TN,
            Y_FP=FP,
            Y_FN=FN,
            loss_bce=loss_bce.item(),
            loss_mse=loss_mse.item(),
            loss_adv_gender_race=loss_adv_gender_race.item(),
            total_loss=total_loss,
        )
    )

    log_file.close()

    return avg_loss, avg_accuracy, y_disc_gender, y_disc_race,avg_val_error

def test_model_EO(
    ML_model: AdversarialAutoEncoder,
    loader,
    criterion_bce,
    criterion_mse,
   # criterion_adv,
    batch,
    alpha,
    beta,
    gamma,
    log_filename,
    epoch =1,
):
    _write_header = True
    # avoid writing header multiple times
    if os.path.exists(log_filename):
        _write_header = False

    log_file = open(log_filename, "a")
    writer = DictWriter(
        log_file,
        fieldnames=[
            "epoch",
            "avg_loss",
            "avg_accuracy",
            "y_disc_gender",
            "y_disc_race",
            "y_disc_gender_white",
            "y_disc_gender_non_white",
            "y_disc_female_race",
            "y_disc_male_race",
            "y_disc_avg_race_gender",
            "y_disc_abs_race_gender",
            "Y_TP", 
            "Y_TN", 
            "Y_FP", 
            "Y_FN", 
            "loss_bce",
            "loss_mse",
            "loss_adv_gender_race",
            "total_loss",
        ],
        dialect="unix",
    )
    if _write_header:
        writer.writeheader()

    ML_model.eval()

    total_accuracy = 0.0
    total_loss = 0.0

    correct = 0
    total = 0

    gen_1_Y0 = gen_1_Y1 = gen_0_Y0 = gen_0_Y1 = 0
    pred_Y1_gen_1_Y0 = pred_Y1_gen_1_Y1 = pred_Y1_gen_0_Y0 = pred_Y1_gen_0_Y1 = 0
    race_1_Y0 = race_1_Y1 = race_0_Y0 = race_0_Y1 = 0
    pred_Y1_race_1_Y0 = pred_Y1_race_1_Y1 = pred_Y1_race_0_Y0 = pred_Y1_race_0_Y1 = 0

    gen_1_race_0_Y0 = gen_1_race_0_Y1 = gen_0_race_0_Y0 = gen_0_race_0_Y1 = 0
    pred_Y1_gen_1_race_0_Y0 = pred_Y1_gen_1_race_0_Y1 = pred_Y1_gen_0_race_0_Y0 = (
        pred_Y1_gen_0_race_0_Y1
    ) = 0
    gen_1_race_1_Y0 = gen_1_race_1_Y1 = gen_0_race_1_Y0 = gen_0_race_1_Y1 = 0
    pred_Y1_gen_1_race_1_Y0 = pred_Y1_gen_1_race_1_Y1 = pred_Y1_gen_0_race_1_Y0 = (
        pred_Y1_gen_0_race_1_Y1
    ) = 0
    race_1_gen_0_Y0 = race_1_gen_0_Y1 = race_0_gen_0_Y0 = race_0_gen_0_Y1 = 0
    pred_Y1_race_1_gen_0_Y0 = pred_Y1_race_1_gen_0_Y1 = pred_Y1_race_0_gen_0_Y0 = (
        pred_Y1_race_0_gen_0_Y1
    ) = 0
    race_1_gen_1_Y0 = race_1_gen_1_Y1 = race_0_gen_1_Y0 = race_0_gen_1_Y1 = 0
    pred_Y1_race_1_gen_1_Y0 = pred_Y1_race_1_gen_1_Y1 = pred_Y1_race_0_gen_1_Y0 = (
        pred_Y1_race_0_gen_1_Y1
    ) = 0
    TP = TN = FP = FN = 0
    val_error=0.0
    # --------------------------
    criterion_adv = CustomFarthestLoss() 
    with torch.inference_mode():
        for i, (X, Y, gender_race) in enumerate(loader):
            X_hat, Y_hat, gender_race_hat = ML_model(X)
            gender_race=gender_race.long()
            loss_bce = criterion_bce(Y_hat.squeeze(), Y)
            loss_mse = criterion_mse(X_hat, X)
            loss_adv_gender_race = criterion_adv(gender_race_hat,gender_race)
   

 
            loss_adv= beta * loss_adv_gender_race
            loss_encoder_decoder_predictor=alpha * loss_mse + gamma * loss_bce

            fpr, fnr = compute_error_rates_multi(Y, Y_hat, gender_race)
            #eo_loss_value = (beta/2)* (1-eo_loss_multi(fpr, fnr))
            loss_adv= loss_adv #+eo_loss_value

            loss = (
                  loss_encoder_decoder_predictor
                + loss_adv
                   )
            val_error+= (beta * loss_adv_gender_race +gamma * loss_bce).item()
            total_loss += loss.item()

            # ===================================================================
            # Calculate accuracy
            predicted = (Y_hat > 0.5).float()  # Threshold predictions
            total_accuracy += (predicted.squeeze() == Y).float()
            #total_accuracy += accuracy_calculator_per_batch(
            #   predicted_label=Y_hat, label=Y
            #)

            # ===================================================================
            # MHK not sure what y_acc means at all in this context!


            T_gen = single_disc_EO_per_batch_multi(predicted_label=Y_hat, label=Y, sf=gender_race,gen=True)
            gen_1_Y0 += T_gen[0]
            gen_1_Y1 += T_gen[1]
            gen_0_Y0 += T_gen[2]
            gen_0_Y1 += T_gen[3]

            pred_Y1_gen_1_Y0 += T_gen[4]
            pred_Y1_gen_1_Y1 += T_gen[5]
            pred_Y1_gen_0_Y0 += T_gen[6]
            pred_Y1_gen_0_Y1 += T_gen[7]

            T_race = single_disc_EO_per_batch_multi(predicted_label=Y_hat, label=Y, sf=gender_race,gen=False)
            race_1_Y0 += T_race[0]
            race_1_Y1 += T_race[1]
            race_0_Y0 += T_race[2]
            race_0_Y1 += T_race[3]

            pred_Y1_race_1_Y0 += T_race[4]
            pred_Y1_race_1_Y1 += T_race[5]
            pred_Y1_race_0_Y0 += T_race[6]
            pred_Y1_race_0_Y1 += T_race[7]

            T_double = double_disc_EO_per_batch_multi(
                predicted_label=Y_hat, label=Y, sf=gender_race)
            gen_1_race_0_Y0 += T_double[0]
            gen_1_race_0_Y1 += T_double[1]
            gen_0_race_0_Y0 += T_double[2]
            gen_0_race_0_Y1 += T_double[3]

            gen_1_race_1_Y0 += T_double[4]
            gen_1_race_1_Y1 += T_double[5]
            gen_0_race_1_Y0 += T_double[6]
            gen_0_race_1_Y1 += T_double[7]

            pred_Y1_gen_1_race_0_Y0 += T_double[8]
            pred_Y1_gen_1_race_0_Y1 += T_double[9]
            pred_Y1_gen_0_race_0_Y0 += T_double[10]
            pred_Y1_gen_0_race_0_Y1 += T_double[11]

            pred_Y1_gen_1_race_1_Y0 += T_double[12]
            pred_Y1_gen_1_race_1_Y1 += T_double[13]
            pred_Y1_gen_0_race_1_Y0 += T_double[14]
            pred_Y1_gen_0_race_1_Y1 += T_double[15]

            #-----------------TP TN FP FN---------------------
            confusion = confusion_matrix_metrics(y_true=Y,y_pred=Y_hat)#,class_names=['0', '1'])
            TP += confusion.get("TP")
            TN += confusion.get("TN")
            FP += confusion.get("FP")
            FN += confusion.get("FN")

    race_1_gen_0_Y0 = gen_0_race_1_Y0
    race_1_gen_0_Y1 = gen_0_race_1_Y1
    race_0_gen_0_Y0 = gen_0_race_0_Y0
    race_0_gen_0_Y1 = gen_0_race_0_Y1

    pred_Y1_race_1_gen_0_Y0 = pred_Y1_gen_0_race_1_Y0
    pred_Y1_race_0_gen_0_Y0 = pred_Y1_gen_0_race_0_Y0
    pred_Y1_race_1_gen_0_Y1 = pred_Y1_gen_0_race_1_Y1
    pred_Y1_race_0_gen_0_Y1 = pred_Y1_gen_0_race_0_Y1

    race_1_gen_1_Y0 = gen_1_race_1_Y0  # MHK
    race_1_gen_1_Y1 = gen_1_race_1_Y1  # MHK
    race_0_gen_1_Y0 = gen_1_race_0_Y0  # MHK
    race_0_gen_1_Y1 = gen_1_race_0_Y1  # MHK

    pred_Y1_race_1_gen_1_Y0 = pred_Y1_gen_1_race_1_Y0  # MHK
    pred_Y1_race_0_gen_1_Y0 = pred_Y1_gen_1_race_0_Y0  # MHK
    pred_Y1_race_1_gen_1_Y1 = pred_Y1_gen_1_race_1_Y1  # MHK
    pred_Y1_race_0_gen_1_Y1 = pred_Y1_gen_1_race_0_Y1  # MHK

    y_disc_gender = (
        abs(pred_Y1_gen_1_Y0 / gen_1_Y0 - pred_Y1_gen_0_Y0 / gen_0_Y0)
        + abs(pred_Y1_gen_1_Y1 / gen_1_Y1 - pred_Y1_gen_0_Y1 / gen_0_Y1)
    ) / 2

    y_disc_race = (
        abs(pred_Y1_race_1_Y0 / race_1_Y0 - pred_Y1_race_0_Y0 / race_0_Y0)
        + abs(pred_Y1_race_1_Y1 / race_1_Y1 - pred_Y1_race_0_Y1 / race_0_Y1)
    ) / 2

    # # finding y_disc for double sensitive features
    y_disc_gender_race_non_w = (
        abs(
            pred_Y1_gen_1_race_0_Y0 / gen_1_race_0_Y0
            - pred_Y1_gen_0_race_0_Y0 / gen_0_race_0_Y0
        )
        + abs(
            pred_Y1_gen_1_race_0_Y1 / gen_1_race_0_Y1
            - pred_Y1_gen_0_race_0_Y1 / gen_0_race_0_Y1
        )
    ) / 2

    y_disc_gender_race_w = (
        abs(
            pred_Y1_gen_1_race_1_Y0 / gen_1_race_1_Y0
            - pred_Y1_gen_0_race_1_Y0 / gen_0_race_1_Y0
        )
        + abs(
            pred_Y1_gen_1_race_1_Y1 / gen_1_race_1_Y1
            - pred_Y1_gen_0_race_1_Y1 / gen_0_race_1_Y1
        )
    ) / 2

    y_disc_race_gender_male = (
        abs(
            pred_Y1_race_1_gen_0_Y0 / race_1_gen_0_Y0
            - pred_Y1_race_0_gen_0_Y0 / race_0_gen_0_Y0
        )
        + abs(
            pred_Y1_race_1_gen_0_Y1 / race_1_gen_0_Y1
            - pred_Y1_race_0_gen_0_Y1 / race_0_gen_0_Y1
        )
    ) / 2

    y_disc_race_gender_female = (
        abs(
            pred_Y1_race_1_gen_1_Y0 / race_1_gen_1_Y0
            - pred_Y1_race_0_gen_1_Y0 / race_0_gen_1_Y0
        )
        + abs(
            pred_Y1_race_1_gen_1_Y1 / race_1_gen_1_Y1
            - pred_Y1_race_0_gen_1_Y1 / race_0_gen_1_Y1
        )
    ) / 2

    y_disc_abs_race_gender = (
        y_disc_gender_race_non_w
        + y_disc_gender_race_w
        + y_disc_race_gender_male
        + y_disc_race_gender_female
    )
    y_disc_avg_race_gender = y_disc_abs_race_gender / 4

    # ===================================================================

    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy.mean().item() / len(loader)
    avg_val_error =val_error/len(loader)

    writer.writerow(
        dict(
            epoch=epoch,
            avg_loss=avg_loss,
            avg_accuracy=avg_accuracy,
            y_disc_gender=y_disc_gender.item(),
            y_disc_race=y_disc_race.item(),
            y_disc_gender_white=y_disc_gender_race_w.item(),
            y_disc_gender_non_white=y_disc_gender_race_non_w.item(),
            y_disc_female_race=y_disc_race_gender_female.item(),
            y_disc_male_race=y_disc_race_gender_male.item(),
            y_disc_avg_race_gender=y_disc_avg_race_gender.item(),
            y_disc_abs_race_gender=y_disc_abs_race_gender.item(),
            Y_TP=TP,
            Y_TN=TN,
            Y_FP=FP,
            Y_FN=FN,
            loss_bce=loss_bce.item(),
            loss_mse=loss_mse.item(),
            loss_adv_gender_race=loss_adv_gender_race.item(),
            total_loss=total_loss,
        )
    )

    log_file.close()

    return avg_loss, avg_accuracy, y_disc_gender, y_disc_race,avg_val_error


