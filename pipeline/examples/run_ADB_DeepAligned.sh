#!/usr/bin bash

# for seed in 0 1 2 3 4 5
# do
#     for dataset in 'clinc' 'banking' 'snips' 'stackoverflow'
#     do
#         for known_cls_ratio in 0.25 0.5 0.75
#         do
#             for labeled_ratio in 0.5 1.0
#             do 
#                 python run.py \
#                 --dataset $dataset \
#                 --detection_method 'ADB' \
#                 --discovery_method 'DeepAligned' \
#                 --known_cls_ratio $known_cls_ratio \
#                 --labeled_ratio $labeled_ratio \
#                 --seed $seed \
#                 --detection_train \
#                 --discovery_train \
#                 --detection_save_model \
#                 --discovery_save_model \
#                 --detection_save_results \
#                 --discovery_save_results \

#             done
#         done
#     done
# done

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
                --seed $seed \
                --train \
                --backbone 'bert' \
                --save_model \
                --save_results \
                --results_file_name 'results_ADB.csv' 
            done
        done
    done
done