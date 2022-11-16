
root="Butterfly/train"
save_folder="test_results_2"

P_dataset=${root}"/Positive"
N_dataset=${root}"/Negative"
U_dataset=${root}"/Unlabel"

nohup python train.py \
    --PNU --P_dataset ${P_dataset} --N_dataset ${N_dataset} --U_dataset ${U_dataset} --save_dir ${save_folder} \
    --P_n 200 --N_n 200 --U_n 2000 \
    --model ResNet18 --prior 0.2 --eta 0.1 --epochs 200 --batchsize 128 --lr 0.000001 \
    > log.txt 

data="Butterfly/test"
python segmentation.py  \
    --dataset ${data} --model ${save_folder}/model.h5 \
    --patch_size 31 --stride 5 --save_dir ${save_folder}