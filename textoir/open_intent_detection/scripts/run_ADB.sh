#!/usr/bin bash

for seed in 0 1 2 3 4 5 6 7 8 9 10
do
    for dataset in 'banking' 'oos' 'stackoverflow'
    do
        for known_cls_ratio in 0.25 0.5 0.75
        do
            for labeled_ratio in 1.0
            do 
                python run_detect.py \
                --dataset $dataset \
                --method 'ADB' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --num_train_epochs 100 \
                --gpu_id '1' \
                --train_detect \
                --save_detect \
                --freeze_bert_parameters

            done
        done
    done
done