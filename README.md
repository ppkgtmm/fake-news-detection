# Fake news detection

Fake news is a serious concern due to its ability to cause destructive impacts on society and nation. Figuring out authenticity of news is crucial before making decisions that can affect people around us. Due to technological advancement, people can be made more convinient by having a tool that automates detection of fake news and this sort of automation can be achieved using machine learning and text mining. Therefore, this project has been initiated to use news data for authenticity prediction as either REAL or FAKE

## General info
- Repo for storing source code of fake news detection end-to-end machine learning project which involves process from data cleaning, visualization,
model training, parameter tuning to model inference
- This project is a portfolio project of owner and is not associated with any courses or institutions

## Overview
<img src="https://user-images.githubusercontent.com/57994731/158847567-2ecf9a20-f8ba-4bbe-a953-941c392288d4.png" />

- Finding of data visualization step can be found [here](https://github.com/ppkgtmm/fake-news-detection/blob/main/visualization/README.md). Visualizations are related to label (REAL or FAKE) distribution, news subject distribution, common words in news, news word count and news average word length
- In modeling part, for classification 2 algorithms were used namely Logistic Regression i.e. GLM of binomial family and Multinomial Naive Bayes
- Both algorithms' predictive power was compared using AUC score and the [result](https://github.com/ppkgtmm/fake-news-detection/blob/main/outputs/2022-03-19/10-19-45/run_modeling.log) is below

```txt
[2022-03-19 10:19:45,941][__main__][INFO] - Config param validation successful
[2022-03-19 10:19:45,941][__main__][INFO] - Begin modeling process
[2022-03-19 10:20:50,652][modeling.training][INFO] - Logistic regression test AUC score : 0.9882990562678062
[2022-03-19 10:21:15,301][modeling.training][INFO] - Multinomial NB test AUC score : 0.9468460648148149
[2022-03-19 10:21:15,301][__main__][INFO] - End modeling process
```

- Finally, the Logistic Regression algorithm was selected and used for further tuning

## Set up
1. Install [Python 3.8 or above](https://www.python.org/downloads/)
2. Install [Java 8 or above](https://www.oracle.com/java/technologies/downloads/)
3. In the project directory, run below to create a new virtual environment
```sh
python3 -m venv <path-to-virtual-environment>
```
4. Run below to activate the virtual enviroment created
```sh
source <path-to-virtual-environment>/bin/activate
```
5. Run below to install required dependencies
```sh
pip3 install -r requirements.txt
```
6. You are ready to go !

## Run
- Each of the steps uses a YAML configuration file stored in config folder of project root directory
- The steps below assumes that you are in the root directory of project
### Run preprocessing
```sh
python3 run_preprocessing.py
```
- By default, preprocessed version of dataset is saved to data directory with prep suffix

### Create visualizations
```sh
python3 run_visualization.py
```
- By default, all visualizations are saved to visualization/outputs directory

### Do modeling
```sh
python3 run_modeling.py
```
- Modeling part require 8 GB of RAM by default but the limit is configurable by editing driver_memory in config/modeling.yaml

### Tune parameters
```sh
python3 run_tuning.py
```
- By default, best output model is saved to modeling/outputs directory. As well, parameter performance summary is stored to modeling/outputs directory as a CSV file
- Tuning part also require 8 GB of RAM by default but the limit is configurable by editing driver_memory in config/modeling.yaml

### Serve model
```sh
uvicorn app:app --reload
```
- API endpoint /predict receives texts, process them, perform prediction and return the prediction results
- First request to /predict endpoint might be slow due to spark model set up (deserialization)

## Sample work
- I picked a fake news followed by a real news to send for prediction
![image](https://user-images.githubusercontent.com/57994731/159123995-4a1aba6e-85ed-4b8b-aea9-17942d356ce9.png)
- Prediction results are below; the first item of probability values corresponds to the REAL class while the last item is likeliness of news for being FAKE
![image](https://user-images.githubusercontent.com/57994731/159124031-e8868f7d-7404-4a08-8861-0ca763ae9564.png)


