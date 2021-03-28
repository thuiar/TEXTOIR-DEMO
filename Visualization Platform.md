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

* **Open Intent Detection Baselines**

[Discovering New Intents with Deep Aligned Clustering](https://github.com/thuiar/DeepAligned-Clustering)

[Deep Unknown Intent Detection with Margin Loss](https://github.com/thuiar/DeepUnkID)

[DOC: Deep Open Classification of Text Documents](https://www.aclweb.org/anthology/D17-1314.pdf)

[A baseline for detecting misclassified and out-of-distribution examples in neural networks](https://arxiv.org/pdf/1610.02136.pdf)

[Towards open set deep networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)

* **Open Intent Discovery**

Open Intent Discovery module integrates the current mainstream 10 baselines.Among them, there were 5 semi-supervised methods and 5 unsupervised methods. By defining the interface, a good code format is formed. If you want to add baseline, you can add your own baseline refer to our code format. On the other hand, in order to provide data support to the visual platform, the 10 baselines need to save the intermediate results of the model running. The following is the introduction of baseline and json format.

* **Open Intent Discovery Semi-Supervised Baselines**

[Discovering New Intents with Deep Aligned Clustering](https://github.com/thuiar/DeepAligned-Clustering)

[Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement](https://github.com/thuiar/CDAC-plus)

[Learning to discover novel visual categories via deep transfer clustering](https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_Learning_to_Discover_Novel_Visual_Categories_via_Deep_Transfer_Clustering_ICCV_2019_paper.pdf)

[Multi-class classification without multi-class labels](https://arxiv.org/pdf/1901.00544.pdf)

[Learning to cluster in order to transfer across domains and tasks](https://arxiv.org/pdf/1711.10125.pdf)

* **Open Intent Discovery Unsupervised Baselines**

[Some methods for classification and analysis of multivariate observations](https://www.cs.cmu.edu/~bhiksha/courses/mlsp.fall2010/class14/macqueen.pdf)

[Agglomerative clustering using the concept of mutual nearest neighbourhood.](https://www.sciencedirect.com/science/article/abs/pii/0031320378900183)

[Unsupervised deep embedding for clustering analysis](http://proceedings.mlr.press/v48/xieb16.pdf)

[Multi-class classification without multi-class labels](https://arxiv.org/pdf/1901.00544.pdf)

[Towards k-means-friendly spaces: Simultaneous deep learning and clustering](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf)

## :hammer_and_wrench: Install
```
PyTorch                    
pytorch_pretrained_bert
matplotlib
sklearn
nltk
gensim
seaborn
tensorflow-gpu
keras
wordcloud
keybert
```













