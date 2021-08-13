#!/usr/bin bash

for seed in 0
do
    for dataset in 'clinc' 'banking' 'stackoverflow' 'snips'
    do
        for known_cls_ratio in 0.25 0.5 0.75
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
                --backbone 'bert' \
                --save_results \
                --save_model \
                --save_frontend_results \
                --exp_name 'ADB_DeepAligned' \
                --results_file_name 'results_ADB.csv' 

                python ../run.py \
                --type 'Discovery' \
                --dataset $dataset \
                --method 'DeepAligned' \
                --setting 'semi_supervised' \
                --config_file_name 'DeepAligned.py' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert' \
                --save_model \
                --save_results \
                --save_frontend_results \
                --exp_name 'ADB_DeepAligned' \
                --results_file_name 'results_DeepAligned.csv'   

                python ../run.py \
                --type 'Pipeline' \
                --seed $seed \
                --dataset $dataset \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --method 'ADB+DeepAligned' \
                --save_results \
                --save_frontend_results \
                --results_file_name 'results_ADB_DeepAligned.csv'
            done
        done
    done
done