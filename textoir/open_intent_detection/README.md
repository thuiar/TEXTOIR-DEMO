## Introduction
This is a toolkit for text open intent detection.

### Usage
### Parameters
dataset: banking | oos | stackoverflow | snips   
known_cls_ratio: 0.25 | 0.5 | 0.75 | 1.0 (default)  
labeled_ratio: 0.2 | 0.4 | 0.6 | 0.8 | 1.0 (default)  
method: ADB  
backbone: bert  
#### An Example
python run.py --dataset banking --known_cls_ratio 0.25 --labeled_ratio 0.1 --method ADB --backbone bert

aoligei!!!