# TEXTOIR: An Integrated and Visualized Platform for Text Open Intent Recognition

## :pushpin: What is TEXTOIR
**TEXOIR** is the first integrated and visualized platform for text Open Intent Recognition. This repo contains a pipeline framework to 

We divide OIR into two categories, open intent detection and open intent discovery. Though both of them are both essential modules for OIR, they have different characteristics and implementation methods. However, there is little related work for integrating these two modules. Although some toolkits are developed for Named Entity Recognition, Sentiment Analysis, there is little related work for intent recognition, not even in open-world settings. Therefore, we build a text open intent recognition platform to fill the blank.

## :couple: Audiences

* **Researchers:** With the help of **TEXTOIR**, you can quickly reproduce the open intention recognition models and analyze the effect of the comparison models.
* **New to Open Intent Recognition:** With the help of **TEXTOIR**, you can fully understand the progress in the field of open intent, and process visualization also helps you understand the model in depth.

## :house:  Overview
Our system consists of two parts, which are ***Visualization Platform*** and ***TEXTOIR Toolkit*** respectively. 
* **TEXTOIR** provides a series of convenient visualized tools for data and model management, training and evaluation, These functions are integrated as Visualization platform. Visualization Platform consists of four parts, which are Dataset Management,Open Intent Detection, Open Intent Discovery and Open Intent Recognition respectively.
* **TEXTOIR** integrates five state-of-the-art models for open intent detection, and ten competitive models for open intent discovery respectively. Moreover, we propose a unified open intent recognition (OIR) framework, which connects the two modules in a pipeline scheme and achieves multiple model combination. These functions are called TEXTOIR Toolkit .

## :chart_with_upwards_trend: Visualization Platform
![image](https://user-images.githubusercontent.com/37832030/114356425-294fc400-9ba3-11eb-8f6f-16e60ca7f9c8.png)

* As you can see, Dataset Management manages your dataset and you can do some basic operations, such as adding, deleting, modifying and searching.
* With Open Intent Detection and Open Intent Discovery you can not only manage the new intent detection models but also to train , test models. At the same time, you can analyze the models' effect.
* We connect the open intent detection and intent discovery modules in a pipeline scheme and achieves multiple model combination. Based on the pipeline scheme, the application of Open Intent Recognition is realized.
### 	:world_map: Working Directory
```
.
├── thedataset       
├── detection   
├── discovery    
├── annotation   
├── static     
│   ├── img  
│   ├── jsons 
│   ├── lib   
│   ├── log   
│   └── video  
├── templates   
│   ├── annotation
│   ├── detection
│   ├── discovery
│   ├── home-page.html
│   ├── index.html
│   ├── system_information
│   └── thedataset
├── textoir 
├── manage.py  
├── db.sqlite3
└── README.md
```
### :loudspeaker: How to use
* **Dataset Management:** The data management module stores the popular open intent identification public datasets. Similarly, to serve the data annotation function, users can upload the dataset that needs to be annotated in this module.
![image](https://user-images.githubusercontent.com/37832030/114355420-096bd080-9ba2-11eb-9119-1f8ea90a608c.png)

* **Open Intent Detection:** Open Intent Detection module provides four basic functions, including Model Management, Model Training, Model Testing, and Model Analysis.
![image](https://user-images.githubusercontent.com/37832030/114355474-1983b000-9ba2-11eb-8f30-527dce36a426.png)

* **Open Intent Discovery:** Open Intent Detection module provides four basic functions, including Model Management, Model Training, Model Testing, and Model Analysis.
![image](https://user-images.githubusercontent.com/37832030/114355539-2accbc80-9ba2-11eb-8fc6-e293be402818.png)

* **Open Intent Recognition:**  Open Intent Recognition module provides data label recommendation function. This module integrates new intent detection and new intent discovery to recommend tags for unlabeled data.
![image](https://user-images.githubusercontent.com/37832030/114355591-3c15c900-9ba2-11eb-8ffc-4257fc70fd98.png)

## :toolbox: TEXTOIR Toolkit
TEXTOIR Toolkit consists of four parts, which are Dialogue Texts,Open Intent Detection, Open Intent Discovery and Open Intent Recognition respectively.There are two main modules, open intent detection and open intent discovery, which integrates most of the state-of-the-art algorithms respectively.Moreover, we propose a unified open intent recognition (OIR) framework,which connects the two modules in a pipeline scheme and achieves multiple model combination.
![image](https://user-images.githubusercontent.com/37832030/114355848-88610900-9ba2-11eb-9586-c9931eabe64b.png)

### 	:world_map: Working Directory
```
.
├── data  
│   ├── banking
│   ├── snips
│   ├── atis
│   ├── dbpedia
│   ├── clinc
│   └── stackoverflow
├── open_intent_detection  
│   ├── Backbone.py
│   ├── dataloader.py
│   ├── init_parameters.py
│   ├── utils.py
│   ├── pretrain.py
│   ├── methods
│   └── README.md
├── open_intent_discovery  
│   ├── Backbone.py
│   ├── dataloader.py
│   ├── init_parameters.py
│   ├── utils.py
│   ├── methods
│   └── README.md
├── pipeline
│   ├── dataloader.py
│   ├── init_parameters.py
│   ├── manager.py
│   ├── utils.py
│   ├── pipe_results
│   └── README.md
├── pipe.py
├── run_detect.py 
├── run_discover.py 
└── Tutorial.md
```
### :loudspeaker: How to use
* **Open Intent Detection module** integrates the current mainstream 5 baselines. By defining the interface, a good code format is formed. If you want to add baseline, you can add your own baseline refer to our code format. On the other hand, in order to provide data support to the visual platform, the 5 baselines need to save the intermediate results of the model running. 
* **Open Intent Discovery module** integrates the current mainstream 10 baselines.Among them, there were 5 semi-supervised methods and 5 unsupervised methods. By defining the interface, a good code format is formed. If you want to add baseline, you can add your own baseline refer to our code format. On the other hand, in order to provide data support to the visual platform, the 10 baselines need to save the intermediate results of the model running. 
* **Open Intent Recognition**The two modules of open intent detection and dis-covery are closely related. However, there lackan overall framework to successively invoke the two modules for both identifying known intentsand discovering open intents. TEXTOIR addressesthis issue with the proposed overall framework.

## :file_folder: Our works 
:page_facing_up:[Discovering New Intents with Deep Aligned Clustering (Accepted by AAAI2021)](https://github.com/thuiar/DeepAligned-Clustering)

:page_facing_up:[Deep Open Intent Classification with Adaptive Decision Boundary (Accepted by AAAI2021)](https://github.com/thuiar/Adaptive-Decision-Boundary)

:page_facing_up:[Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement (Accepted by AAAI2020)](https://github.com/thuiar/CDAC-plus)

:page_facing_up:[Deep Unknown Intent Detection with Margin Loss (Accepted by ACL2019)](https://github.com/thuiar/DeepUnkID)

## :hammer_and_wrench:  Install
```
Django
PyMySQL
pandas
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


  
