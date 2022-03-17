# Fake news detection

The fake news is a serious concern due to its ability to cause destructive impacts on society and nation. Figuring out authenticity of news is crucial before making decisions that can affect people around us. Due to technological advancement, people can be made more convinient by having a tool that automates detection of fake news and this sort of automation can be achieved using machine learning and text mining. Therefore, this project has been initiated to use news data for authenticity prediction as either REAL or FAKE

## General info
- Repo for storing source code of fake news detection end-to-end machine learning project which involves process from data cleaning, visualization, 
model training, parameter tuning to model inference
- This project is a portfolio project of owner and is not associated with any courses or institutions

## Overview
<img src="https://user-images.githubusercontent.com/57994731/158847567-2ecf9a20-f8ba-4bbe-a953-941c392288d4.png" />

### Set up
1. Install [Python 3.6 or above](https://www.python.org/downloads/)
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


