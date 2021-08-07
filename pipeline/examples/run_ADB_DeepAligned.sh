#!/usr/bin bash

for seed in 0
do
    for dataset in 'banking'
    do
        for known_cls_ratio in 0.25
        do
            for labeled_ratio in 1.0
            do 
                python ../run.py \
                --type 'Detection' \
                --dataset $dataset \
                --method 'ADB' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --config_file_name 'ADB.py' \
                --seed $seed \
                --train \
                --backbone 'bert' \
                --save_model \
                --save_results \
                --results_file_name 'results_ADB.csv' 

                python ../run.py \
                --type 'Discovery' \
                --dataset $dataset \
                --method 'DeepAligned' \
                --setting 'semi_supervised' \
                --config_file_name 'DeepAligned' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --train \
                --backbone 'bert' \
                --save_model \
                --save_results \
                --results_file_name 'results_DeepAligned.csv'   

                python ../run.py \
                --type 'Pipeline' \
                --seed $seed 
            done
        done
    done
done