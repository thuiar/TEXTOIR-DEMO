# TEXTOIR HandBook

## :pushpin: What is TEXTOIR
**TEXOIR** is the first integrated and visualized platform for text Open Intent Recognition(OIR). 

We divide OIR into two categories, open intent detection and open intent discovery. Though both of them are both essential modules for OIR, they have different characteristics and implementation methods. However, there is little related work for integrating these two modules. Although some toolkits are developed for Named Entity Recognition, Sentiment Analysis, there is little related work for intent recognition, not even in open-world settings. Therefore, we build a text open intent recognition platform to fill the blank. The demonstration video is available at [there.](https://github.com/XTenLee/TEXTOIR)

## :couple: Audiences

* **Researchers:** With the help of **TEXTOIR**, you can quickly reproduce the open intention recognition models and analyze the effect of the comparison models.
* **New to Open Intent Recognition:** With the help of **TEXTOIR**, you can fully understand the progress in the field of open intent, and process visualization also helps you understand the model in depth.

## :house:  Overview
Our system consists of four parts, which are *Dataset Management*,*Open Intent Detection*, *Open Intent Discovery* and *Data Annotation* respectively. 
* As you can see, *Dataset Management* manages your dataset and you can do some basic operations, such as adding, deleting, modifying and searching.
* With *Open Intent Detection* and *Open Intent Discovery* you can not only manage the new intent detection models but also to train , test models. At the same time, you can analyze the models' effect. 
* We connect the open intent detection and intent discovery modules in a pipeline scheme and achieves multiple model combination. Based on the pipeline scheme, the application of *Data Annotation* is realized.

## :loudspeaker: How to use
![image](https://github.com/XTenLee/TEXTOIR/blob/main/image/handbook.png)
* **Dataset Management:**
The data management module stores the popular open intent identification public data set. Similarly, to serve the data annotation function, users can upload the data set that needs to be annotated in this module.

1. Information of data
Dataset management displays the datasets stored in TEXTOIR, and users can filter the datasets of interest through the search box at the top. Details of  the data set can be viewed in "Details".
Besides, TEXTOIR provides data modification and data deletion capabilities. It should be noted that users are only allowed to delete their uploaded datasets. System datasets cannot be modified.
2. Add the data
TEXTOIR offers a data tag recommendation feature that allows you to upload local data with the "Add" button. Please note that the upload conforms to the system specification.

* **Open Intent Detection:**
Open Intent Detection module provides four basic functions, including Model Management, Model Training, Model Testing, and Model Analysis.
1. Model Management：This module collects the classic baseline and the current SOTA model of the Open Intent Detection. And the user can view information about the model.
2. Model Training：In this section, users can view the run-log, as well as the training parameters. Users can train the model by adding. In the model-training page, the user can select the model, select the data set and set the annotation scale, and know the intention scale.And the corresponding training parameters can be set according to different models. After the successful operation, the corresponding record appears on the operation record page. If the user wants to abort the training, select "Stop".
3. Model Testing：This module shows the results of the model run. First, error analysis is provided for a single model. Based on the user's selection, the results of the model's correct and wrong judgments for each category of intent are presented.Furthermore, the result analysis of the model judgment error is analyzed in detail, which provides data support for the relationship between improving the model and exploring the intention. Secondly, to better observe the effect of intention ratio and labeled ratio, TEXTOIR uses line diagrams to show the influence of the two ratios on the model effect.
4. Model Analysis：This module uses sample examples provided by Textoir to visualize the underlying principles of analysis model judgment. For the open intention detection model, there are two basic solutions, the probability threshold method and the decision boundary method. TEXTOIR provides graphical representations of each of these two types of models.
* **Open Intent Discovery:**
Open Intent Detection module provides four basic functions, including Model Management, Model Training, Model Testing, and Model Analysis.
1. Model Management：This module collects the classic baseline and the current SOTA model of the Open Intent Discovery. And the user can view information about the model.
2. Model Training：In this section, users can view the run-log, as well as the training parameters. Users can train the model by adding. In the model-training page, the user can select the model, select the data set and set the annotation scale, and know the intention scale.And the corresponding training parameters can be set according to different models. After the successful operation, the corresponding record appears on the operation record page. If the user wants to abort the training, select "Stop".
3. Model Testing：This module shows the results of the model run. First, error analysis is provided for a single model. Based on the user's selection, the results of the model's correct and wrong judgments for each category of intent are presented.Furthermore, the result analysis of the model judgment error is analyzed in detail, which provides data support for the relationship between improving the model and exploring the intention. Second, to better observe the influence of intention ratio and clusters number, Textoir uses line diagrams to show the influence of the two ratios on the model effect.
4. Model Analysis：This module uses sample examples provided by Textoir to visualize the underlying principles of analysis model judgment. For the open intent discovery model, Textoir uses pie charts to show the distribution of intents and display Textoir recommendation tags.
* **Data Annotation:**
Data Annotation module provides data label recommendation function. This module integrates new intent detection and new intent discovery to recommend tags for unlabeled data.
1. Data Preparation, TEXTOIR provides a data upload interface in the data management section so that users can upload their own data sets to be annotated according to their actual needs.
2. Data Annotation, TEXTOIR combines SOTA open intent detection and open intent discovery models to provide a paradigm for open intent identification. On the basis of pipline schema, we integrate the tag recommendation algorithm and form the basic structure of data tag recommendation.In the data annotation page, users select the data set to be annotated and the annotation algorithm to perform data annotation.
3. Label Modified, TEXTOIR gives users the ability to modify labels. The annotation model provided by Textoir will generate four recommendation labels and automatically select the one with the highest probability as a "selected Label". If users are not satisfied with the recommended tags provided by the system, they can modify the tags by clicking the remaining candidate tags.


## :hammer: Install

After updating...

## :question: FAQ

After updating...


  
  
