NUM_hidden_LAYER=1
ALPHA=`awk 'BEGIN{print 0.05}'`
BETA_gender=10
BETA_race=10
GAMMA=1
EPOCH_NUM_stack1=2000
EPOCH_NUM_stack2=2000
SEED=42
BATCH_SIZE=128
INPUT_DIM_LEVEL_1=102
HIDDEN_DIM_LEVEL_1=70
ENCODED_DIM_LEVEL_1=40
HIDDEN_DIM_LEVEL_2=25
ENCODED_DIM_LEVEL_2=8
LR=`awk 'BEGIN{print 0.001}'`
Disc_met=EO
DROP_OUT=`awk 'BEGIN{print 0.5}'`
run_mode=CPU # CPU or GPU
termination_epoch_threshold=50
margin_threshold=`awk 'BEGIN{print 0.05}'`

python run.py --num_layer ${NUM_hidden_LAYER} -a ${ALPHA} --beta ${BETA_gender} -g ${GAMMA} --epoch_num_level_1 ${EPOCH_NUM_stack1} --epoch_num_level_2 ${EPOCH_NUM_stack2} --seed ${SEED} --batch_size ${BATCH_SIZE} --input_dim ${INPUT_DIM_LEVEL_1} --hidden_dim_level_1 ${HIDDEN_DIM_LEVEL_1} --encoded_dim_level_1 ${ENCODED_DIM_LEVEL_1} --hidden_dim_level_2 ${HIDDEN_DIM_LEVEL_2} --encoded_dim_level_2 ${ENCODED_DIM_LEVEL_2}   --discrimination_metric ${Disc_met} --learning_rate ${LR} --dropout_rate ${DROP_OUT} --mode ${run_mode} --termination_epoch_threshold ${termination_epoch_threshold} --margin_threshold ${margin_threshold}

