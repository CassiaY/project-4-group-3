# Heart Disease Risk Analysis  
<font size="3">**Group-Project-4: Team 3**  
**Contributors:** Logan Severson, MaiDao Lor, Kyle Dalton, Cassia Yoon  
**Github link:** https://github.com/CassiaY/project-4-group-3  
**Slide link:** https://www.canva.com/design/DAF2Q4tnnNQ/2TqK71GT2WsZIIw5lsYwtA/view?utm_content=DAF2Q4tnnNQ&utm_campaign=designshare&utm_medium=link&utm_source=editor </font>  

## Project Overview  
For this project, we visualized heart disease data and used machine learning to predict if a patient is at risk of heart disease based on BRFSS data (Behavioral Risk Factor Surveillance System).  

## The Process  
### Part 1: The Data  
[BRFSS](/https://www.cdc.gov/brfss/index.html) data is a yearly phone (landline and cell) survey in the US conducted by the CDC. The survey collects health-related risk behaviors, chronic health conditions, and use of preventive services. The [Diabetes Health Indicators dataset](/https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) we use contains preprocessed data intended to study the relationship between lifestyle and diabetes in the US. For this project, we decided to use this dataset to study heart disease risk instead (the [csv file](/Resources/diabetes_binary_5050split_health_indicators_BRFSS2015.csv) can be found in the Resources folder). Sample of the dataset:  
![sample of the dataset](/readme_imgs/dataset_sample.png)  
The dataset was then stored in a cloud computing service. We used [Amazon Web Services](/https://aws.amazon.com/) using the [S3](/https://aws.amazon.com/s3/) service. In the machine learning code, the data file is accessed via a link that grants temporary public access.  

Seaborn was used to create a scatter plot of all the features vs target (see below). Due to the data having mostly binary data, the scatter plots had limited value.  
![data scatter plots](/readme_imgs/data_scatterplots_all.png)  

When reviewing this binary dataset, there are certain things that we took into consideration:  
1. Response bias: Did all groups in our population have an equal chance of responding?  
2. Representation: are all groups within our population adequately represented in the data?  
3. Data collection method: could the method of data collection such as cellular versus landline influence the responses? For example, younger people might be more likely to respond on a cell phone, while older individuals might prefer landlines. When working with survey data, it's common to encounter non-response bias. This happens when the respondents of the survey differ in meaningful ways from those who did not respond.  

We did some research to see how BRFSS Provides information on the background, data collection and processing, and the statistical, and analytical issues for the combined landline and cell phone data set. We found that:  
1. The behavioral risk factor surveillance system (BRFSS) doesn't have a specific way to include people who don't have access to a telephone and therefore couldn't participate in the survey.  
2. The BRFS uses a statistical technique called raking to adjust their data. This technique helps make the sample more representative of the entire population. If certain groups such as lower income people for example, are underrepresented and the survey responses, raking helps adjust the data to better reflect these groups presence in the overall population.  
4. Even though the BRFSS survey might not reach everyone, **they used statistical techniques (such as raking) to ensure their findings are accurate and as representative of the population as possible.**  


### Part 2: Machine Learning Model  
The code files were created in [Google Colab](/https://colab.google/) and used:  
- [scikit-learn](/https://scikit-learn.org/stable/) - a machine learning library for python  
- [keras-tuner](/https://keras.io/keras_tuner/) - a scalable hyperparameter optimization framework  
- [tensorflow](/https://www.tensorflow.org/) - a library for machine learning and artificial intelligence  
- [pandas](/https://pandas.pydata.org/) - a data analysis and manipulation tool for python  
- [matplotlib](/https://matplotlib.org/) - a library for creating static, animated, and interactive visualizations in Python  
- [seaborn](/https://seaborn.pydata.org/) - a Python data visualization library based on matplotlib  

The file [project_4_group_3_nn.ipynb](/project_4_group_3_nn.ipynb) contains the code used to train the machine learning model to predict heart disease. We used keras-tuner auto-optimization to find the optimal hyperparameters to achieve the highest prediction rating. The maximum number of layers was set to 5. The maximum number of neurons was set to 50, which is between 2-3 times the number of input features.  

![number of input features](/readme_imgs/n_features.png)  

The optimizer was able to achieve an **`accuracy score of 86.11%`** at a **`loss of 32.76%`**.  

![val_accuracy](/readme_imgs/best_model_eval.png)  

The hyperparameters for the best model found were:  
![best model hyperparameters](/readme_imgs/best_hps.png)  

This is the summary of the best model:  
<img src="https://github.com/CassiaY/project-4-group-3/blob/main/readme_imgs/best_model_summary.png?raw=true" width="500">

The best machine learning model was exported in three different verions: [just the weights](/Resources/project-4-group-3-nn-model.h5), the [full model in HDF5 format](/Resources/project-4-group-3-nn-model_full.h5) and the [full model in keras file format](/Resources/project-4-group-3-nn-model_full.keras). These files are found in the 'Resources' folder.  

The model file was then loaded into [project_4_group_3_plt.ipynb](/project_4_group_3_plt.ipynb) to plot the history of the accuracy and loss of the model. The graphs show that the model achieved high accuracy and low loss, which indicates a low amount of prediction errors. However, we can see that there's a persistent gap between the training and validation lines in both graphs.  
| Accuracy                                        | Loss                                    |
| ----------------------------------------------- | --------------------------------------- |
| ![accuracy graph](/readme_imgs/nn_accuracy.png) | ![loss graph](/readme_imgs/nn_loss.png) |

The model was also loaded into [project_4_group_3_weights](/project_4_group_3_weights.ipynb) to obtain the [weights](/readme_imgs/best_model_weights_sample.png).  

We did attempt to extract feature importance from the deep machine learning model, but were not able to succeed in the time given. However, it seems to be possible based on some examples seen online using LIME or SHAP tools. See the Resources section below for links regarding this topic.

Instead, to obtain information regarding feature importance, we trained a random forest model for the data by category and for overall. We also obtained confusion matrices and classificantion reports for each category.  

#### Overall  
![all cm](/readme_imgs/rf_all_cm.png)  
![all f-imp](/readme_imgs/rf_all_feature_importances.png)  
Here we can see that the model can predict who will NOT have heart disease or a heart attack with a precision of 87 percent, recall of 97%, and f1 score of 92%. On the other hand, the precision, recall and f1 scores are significantly lower when trying to predict who WILL have heart disease or a heart attack. Despite this difference, the accuracy score is high at 85%.  
On below that is the list of the top 10 important features sorted from most to least important. The top 5 important features are BMI, sex, education, mental health, and age. It is interesting to note that most of these risk factors are in the health status and demograhic categories. The only exceptions to this are level of physical activity and whether they don't see a doctor due to cost.  

#### Lifestyle  
![lifestyle cm](/readme_imgs/rf_lifestyle_cm.png)  
![lifestyle f-imp](/readme_imgs/rf_lifestyle_feature_importances.png)  
It is important to note that the precision, recall, and f1 scores are 0 when trying to predict 'yes, they'll have heart disease/attack'. But the results do show that not smoking increases your chance of not developing hear disease. the feature importance for this is at 55%, which is pretty high.  

#### Health Status  
![health cm](/readme_imgs/rf_health_cm.png)  
![health f-imp](/readme_imgs/rf_health_feature_importances.png)  
With the health status category, the precision, recall, and f1 scores are higher this time in predicting 'yes, heart disease/attack', suggesting that the risk factors in this category are more important in predicting heart disease. The top feature importances in this category are: BMI, mental health, and general health status.  

#### Healthcare Access  
![healthcare cm](/readme_imgs/rf_healthcare_cm.png)  
![healthcare f-imp](/readme_imgs/rf_healthcare_feature_importances.png)  
With the Healthcare Access category, there weren't a lot of features but seems like getting regular cholesterol checks make one less likely to develop heart disease.  

#### Demographic  
![demographic cm](/readme_imgs/rf_demographics_cm.png)  
![demographic f-imp](/readme_imgs/rf_demographics_feature_importances.png)  
And lastly, with the Demographics category, it seems that sex can play an important role in predicting history of heart disease.  

It was interesting to see that the predictive accuracy for both the keras tuner model and the random forest models were similar, all around 85%. But taking a closer look at the random forest models, we noticed that the models were predicting who will NOT have history of heart disease rather than who will. This may be because this dataset was originally created to predict diabetes rather than heart disease. When we chose this dataset, we thought there would be more of an overlap between the risk factors for diabetes and heart disease but this showed that this was not necessarily the case.  

### Part 3: Visualizations  
To be able to create the visualizations in Tableau, the data needed to be decoded using the [BRFSS Codebook](/https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf), [Survey Questionnaire](/https://www.cdc.gov/brfss/questionnaires/pdf-ques/2015-brfss-questionnaire-12-29-14.pdf), and the variables table from the [UCI dataset website](/https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). For example, the data used codes for respondent's [income](/readme_imgs/questionnaire_income.png), [education level](/readme_imgs/questionnaire_education.png), and [age group](/readme_imgs/codebook_ages.png).  

#### Visualization 1:  
![age/gender graph](/readme_imgs/viz_age_gender.png)  
The goal is to view the relationship between age, gender, and the incidence of heart disease or attacks, while also displaying the total count of respondents within each age group. Tableau was used to create the bar graph and totaled the number of male and female respondents. Each bar is split into two colors, representing whether they have had heart disease or attacks. Light blue represents those that responded ‘no’ and dark blue is ‘yes’. Following the increasing age group, we notice a trend. The dark blue bars that represent yes respondents to heart disease or attacks cases, generally increase. This suggests that heart disease or attacks become more common as age increases. When looking at the colors within each bar, we see another pattern. In some age groups, the section representing males is taller than that for females, indicating that more men in these groups have experienced heart disease or attacks.

#### Visualization 2:  
![categories graph](/readme_imgs/viz_categories.png)  
Based on this graph, we can see that the top behavioral risk factors are: Cholesterol Check, Having Healthcare, Veggies/Fruits, and Physical activities. And other behavioral risk factors include: problems walking, high blood pressure, and heavy alcohol consumption.  

#### Visualization 3:  
![alc/smoking graph](/readme_imgs/viz_alc_smok.png)  
Over time, high blood pressure (hypertension) puts strain on the heart muscle and can lead to cardiovascular disease (CVD), which increases your risk of heart attack and stroke. Chemicals in cigarette smoke cause the blood to thicken and form clots inside veins and arteries. Blockage from a clot can lead to a heart attack and sudden death. While both have negative effects on your body, smoking increases the likelihood of getting any sort of heart disease/attack due to the various chemicals in he smoke that destroy your cells.  

#### Visualization 4:  
![age/bmi graph](/readme_imgs/viz_age_bmi.png)
An individual's body mass index (BMI) isn't a give all tell all when it comes to likelihood of having heart diseases or attacks. It does in fact increase your chances of having one if you have a BMI roughly over 50. It's not a guarantee if you do have a BMI over that amount that you will have one, but it will increase the chances of it happening.  

> Additional visualizations can be found in the folder [readme_imgs](/readme_imgs).

## Project Conclusions:  
In conclusion, we found correlations between different risk factors predicting heart disease or having a heart attack. We determined it is easier to predict the likelihood of who will not have heart disease or a heart attack rather than who will. The risk factors with the highest weights were found to be;
BMI, sex, mental health, and education. For future considerations, we would like to work with more updated data and see different risk factors that can go into predications such as geography/weather, other lifestyle factors, and additional demographics.  

## Resources  
- Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators  
- article discussing guideline for number of hidden layers: https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3#:~:text=Number%20of%20Neurons%20and%20Number%20of%20Layers%20in%20Hidden%20Layer&text=The%20number%20of%20hidden%20neurons,size%20of%20the%20output%20layer
- export entire model (architecture + weights): https://stackoverflow.com/questions/76825216/is-there-a-way-to-save-a-model-with-hyperparameters-and-weights-both-after-hyper
- saving and loading keras model: https://deeplizard.com/learn/video/7n1SpeudvAE#:~:text=To%20do%20so%2C%20we%20first,the%20saved%20model%20on%20disk.&text=We%20can%20verify%20that%20the,get_weights()%20on%20the%20model  
- getting keras model weights:  https://stackoverflow.com/questions/46817085/keras-interpreting-the-output-of-get-weights  
- getting feature importance from a keras model:  https://datascience.stackexchange.com/questions/74661/feature-importance-in-neural-networks
- LIME vs SHAP: https://www.kdnuggets.com/2020/01/explaining-black-box-models-ensemble-deep-learning-lime-shap.html  
- shap usage example: https://colab.research.google.com/github/kweinmeister/notebooks/blob/master/tensorflow-shap-college-debt.ipynb#scrollTo=IytTh1pb0HGN  
- resize github readme img: https://stackoverflow.com/questions/24383700/resize-image-in-the-wiki-of-github-using-markdown  
- how to filter warning in colab: https://copyprogramming.com/howto/python-how-to-disable-warning-on-google-colab  
- S3 bucket granting public access: https://repost.aws/knowledge-center/read-access-objects-s3-bucket  
- how to get number of input features (other than counting): https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html  
- how to plot history of keras model: https://www.kaggle.com/code/danbrice/keras-plot-history-full-report-and-grid-search  

## To Good Health!
If you’re interested, here is a couple links where you can check your risk of heart attacks or strokes in the next 10 year and get some tips on Healthy Lifestyles.  

Calculate Your Heart Age: https://www.nyc.gov/site/doh/health/health-topics/heart-age-calculator.page 

American Heart Associate: https://www.heart.org/en/healthy-living/healthy-lifestyle/lifes-essential-8

## Acknowledgements
We wish to thank our teaching staff:
- Hunter Hollis
- Sam Espe
- Randy Sendek
