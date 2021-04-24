#!/usr/bin bash

for seed in 0 1 2 3
do
    for dataset in 'snips' 'stackoverflow'
    do
        for known_cls_ratio in 0.25 0.5 0.75
        do
            for labeled_ratio in 0.25 0.5 0.75 1.0
            do 
                python pipe.py \
                --dataset $dataset \
                --detect_method 'ADB' \
                --discover_method 'DeepAligned' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --gpu_id '0' \
                --train_detect \
                --save_detect \
                --train_discover \
                --save_discover \

            done
        done
    done
done