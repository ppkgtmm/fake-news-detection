# Fake news detection

The fake news is a serious concern due to its ability to cause destructive impacts on society and nation. Figuring out authenticity of news is crucial before making decisions that can affect people around us. Due to technological advancement, people can be made more convinient by having a tool that automates detection of fake news and this sort of automation can be achieved using machine learning and text mining. Therefore, this project has been initiated to use news data for authenticity prediction as either REAL or FAKE

## General info
- Repo for storing source code of fake news detection end-to-end machine learning project which involves process from data cleaning, visualization,
model training, parameter tuning to model inference
- This project is a portfolio project of owner and is not associated with any courses or institutions

## Overview
<img src="https://user-images.githubusercontent.com/57994731/158847567-2ecf9a20-f8ba-4bbe-a953-941c392288d4.png" />

- In modeling part, for classification 2 algorithms were used namely Logistic Regression i.e. GLM of binomial family and Multinomial Naive Bayes
- Both algorithms' predictive power was compared using AUC score and the [result](https://github.com/ppkgtmm/test-test/blob/main/outputs/2022-03-15/12-15-31/run_modeling.log) is below

```txt
[2022-03-15 12:15:31,997][__main__][INFO] - Config param validation successful
[2022-03-15 12:15:31,997][__main__][INFO] - Begin modeling process
[2022-03-15 12:16:41,879][modeling.training][INFO] - Logistic regression test AUC score : 0.9890527497739088
[2022-03-15 12:17:09,419][modeling.training][INFO] - Multinomial NB test AUC score : 0.9429894896315228
[2022-03-15 12:17:09,420][__main__][INFO] - End modeling process
```

- Finally, the Logistic Regression algorithm was selected and used for further tuning

## Set up
1. Install [Python 3.8 or above](https://www.python.org/downloads/)
2. Run below to create a new virtual environment
```sh
python3 -m venv <path-to-virtual-environment>
```
3. run below to activate the virtual enviroment created
```sh
source <path-to-virtual-environment>/bin/activate
```
4. run below to install required dependencies
```sh
pip3 install -r requirements.txt
```
5. you are ready to go !

## Run
- Each of the steps uses a YAML configuration file stored in config folder of project root directory
- The steps below assumes that you are in the root directory of project
### Run preprocessing
- By default, preprocessed version of dataset is saved to data directory with prep suffix
```sh
python3 run_preprocessing.py
```

### Create Visualizations
- By default, all visualizations are saved to visualization/outputs directory
```sh
python3 run_visualization.py
```

### Do modeling
- Modeling part require 8 GB of RAM by default but the limit is configurable by editing driver_memory in config/modeling.yaml
```sh
python3 run_modeling.py
```
### Tune parameters
- By default, best output model is saved to modeling/outputs directory
- As well, parameter performance is stored to modeling/outputs/lr_tuning_results.csv
- Tuning part also require 8 GB of RAM by default but the limit is configurable by editing driver_memory in config/modeling.yaml
```sh
python3 run_tuning.py
```

### Serve model
- API endpoint /predict receives texts, process them, perform prediction and return the prediction results
- First request to /predict endpoint might be slow due to spark model set up (deserialization)
```sh
uvicorn app:app --reload
```

## Sample work
- I picked a fake news along with a real news to send them for prediction
![image](https://user-images.githubusercontent.com/57994731/158858552-cd4faf9d-9a37-4c37-8b1d-b669250e7ba2.png)
- Prediction results are below; the first item of probability values corresponds to the REAL class while the last item is likeliness of news for being FAKE
![image](https://user-images.githubusercontent.com/57994731/158859077-6ab31639-e658-4e9e-9499-a6cc74a32537.png)

