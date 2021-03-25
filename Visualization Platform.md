# Visualization Platform

## :house_with_garden: Overview


TEXTOIR integrates five state-of-the-art models for open intent detection, and ten competitive models for open intent discovery respectively. Moreover, we propose a unified open intent recognition (OIR) framework, which connects the two modules in a pipeline scheme and achieves multiple model combination. These functions are called TEXTOIR Toolkit.

TEXTOIR Toolkit consists of four parts, which are Dialogue Texts,Open Intent Detection, Open Intent Discovery and Open Intent Recognition respectively.There are two main modules, open intent detection and open intent discovery, which integrates most of the state-of-the-art algorithms respectively.Moreover, we propose a unified open intent recognition (OIR) framework,which connects the two modules in a pipeline scheme and achieves multiple model combination.

![textoir_toolit](https://user-images.githubusercontent.com/37832030/112449266-2a54b900-8d8e-11eb-8dab-8b76ee7ae9fc.jpg)

## :tram: Working Directory

```
                        .
                        ├── data  
                        │   ├── banking
                        │   ├── clinc
                        │   └── stackoverflow
                        ├── open_intent_detection  
                        │   ├── Backbone.py
                        │   ├── dataloader.py
                        │   ├── init_parameters.py
                        │   ├── methods
                        │   ├── pretrain.py
                        │   ├── README.md
                        │   └── utils.py
                        ├── open_intent_discovery  
                        │   ├── Backbone.py
                        │   ├── dataloader.py
                        │   ├── init_parameters.py
                        │   ├── methods
                        │   ├── README.md
                        │   └── utils.py
                        ├── pipeline
                        │   ├── alignment_test.py
                        │   ├── align_test.py
                        │   ├── dataloader.py
                        │   ├── dataloder_.py
                        │   ├── init_parameters.py
                        │   ├── manager.py
                        │   ├── pipe_results
                        │   └── utils.py
                        ├── pipe.py
                        ├── run_detect.py 
                        ├── run_discover.py 
                        └── Tutorial.md
```

## :loudspeaker: How to use
* **Open Intent Detection**

Open Intent Detection module integrates the current mainstream 5 baselines. By defining the interface, a good code format is formed. If you want to add baseline, you can add your own baseline refer to our code format. On the other hand, in order to provide data support to the visual platform, the 5 baselines need to save the intermediate results of the model running. The following is the introduction of baseline and json format.

**Baselines**

[Discovering New Intents with Deep Aligned Clustering](https://github.com/thuiar/DeepAligned-Clustering)

[Deep Unknown Intent Detection with Margin Loss](https://github.com/thuiar/DeepUnkID)

[DOC: Deep Open Classification of Text Documents](https://www.aclweb.org/anthology/D17-1314.pdf)

[A baseline for detecting misclassified and out-of-distribution examples in neural networks](https://arxiv.org/pdf/1610.02136.pdf)

[Towards open set deep networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)





