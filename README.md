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

### Part 2: Machine Learning Model  
The code files were created in [Google Colab](/https://colab.google/) and used:  
- [scikit-learn](/https://scikit-learn.org/stable/) - a machine learning library for python  
- [keras-tuner](/https://keras.io/keras_tuner/) - a scalable hyperparameter optimization framework  
- [tensorflow](/https://www.tensorflow.org/) - a library for machine learning and artificial intelligence  
- [pandas](/https://pandas.pydata.org/) - a data analysis and manipulation tool for python  
- [matplotlib](https://matplotlib.org/) - a library for creating static, animated, and interactive visualizations in Python  

The file [project_4_group_3_nn.ipynb](/project_4_group_3_nn.ipynb) contains the code used to create the machine learning model to predict heart disease. We used keras-tuner auto-optimization to find the optimal hyperparameters to achieve the highest prediction rating. The maximum number of layers was set to 5. The maximum number of epochs was set to 20 to start. The maximum number of neurons was set to 50, which is between 2-3 times the number of input features.  

![number of input features](/readme_imgs/n_features.png)  

The optimizer was able to achieve an **`accuracy score of 86.11%`** at a **`loss of 32.76%`**.  

![val_accuracy](/readme_imgs/best_model_eval.png)  

The hyperparameters for the best model found were:  
![best model hyperparameters](/readme_imgs/best_hps.png)  

The best machine learning model was exported in three different verions: [just the weights](/Resources/project-4-group-3-nn-model.h5), the [full model in HDF5 format](/Resources/project-4-group-3-nn-model_full.h5) and the [full model in keras file format](/Resources/project-4-group-3-nn-model_full.keras). These files are found in the 'Resources' folder.  

The model file was then loaded into [project_4_group_3_plt.ipynb](/project_4_group_3_plt.ipynb) to plot the graph of the accuracy and loss of the model over epochs:  
| Accuracy                                        | Loss                                    |
| ----------------------------------------------- | --------------------------------------- |
| ![accuracy graph](/readme_imgs/nn_accuracy.png) | ![loss graph](/readme_imgs/nn_loss.png) |

### Part 3: Visualizations  
To be able to create the visualizations in Tableau, the data needed to be decoded using the [BRFSS Codebook](/https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf), [Survey Questionnaire](/https://www.cdc.gov/brfss/questionnaires/pdf-ques/2015-brfss-questionnaire-12-29-14.pdf), and the variables table from the [UCI dataset website](/https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). For example, the data used codes for respondent's [income](/readme_imgs/questionnaire_income.png), [education level](/readme_imgs/questionnaire_education.png), and [age group](/readme_imgs/codebook_ages.png).  

Tableau blah blah blah  
Pretty colors 

## Project Conclusions:  


## Resources  
- Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators  
- saving and loading keras model: https://deeplizard.com/learn/video/7n1SpeudvAE#:~:text=To%20do%20so%2C%20we%20first,the%20saved%20model%20on%20disk.&text=We%20can%20verify%20that%20the,get_weights()%20on%20the%20model  
- getting keras model weights:  https://stackoverflow.com/questions/46817085/keras-interpreting-the-output-of-get-weights  

## Acknowledgements
We wish to thank our teaching staff:
- Hunter Hollis
- Sam Espe
- Randy Sendek
