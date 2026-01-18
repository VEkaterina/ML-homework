# Maternal Health Risk Classifier

## 1. Problem Statement  
Maternal health is a critical aspect of public health, as complications during pregnancy and childbirth can have serious consequences for both mothers and babies. Identifying pregnancies that are at higher risk enables healthcare providers to prioritize care and take preventive measures to improve outcomes.

In this project, I aim to develop a machine learning model that predicts the maternal risk level using features such as age, blood pressure, heart rate, and other health indicators. The goal is to provide a tool that assists healthcare professionals in early detection of high-risk pregnancies, potentially improving maternal and fetal health outcomes through timely intervention.

---

## 2. Dataset Description  
For my project, I used the *Maternal Health Risk Data* dataset from Kaggle. Data has been collected from different hospitals, community clinics, maternal health cares through the IoT based risk monitoring system.

**Source**: [Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data)  

The dataset includes features related to age, blood pressure (BP), blood glucose (BS), and heart rate, which have been shown in literature to correlate with pregnancy-related risks.

The dataset can be downloaded direclty form Kaggle following the link above manually or automatically as shown in my Notebook.ipynb. 

---

## 3. Feature Selection & Model Choice  

The features and model selection process is given step-by-step in Notebook.ipynb. To execute this file, download it to your device. You do not need to download the dataset, the code will do this automatically. You only need to ensure that all the packages listed in Requirments are installed.

- **Model Choice**:  
  - I trained and compared multiple models: **Logistic Regression**, **Decision Trees**, and **Random Forests**.  
  - The final model was selected based on its predictive performance, measures by F1-score for each of the classes. **RandomForestClassifier** showed the best perdormance.
  - After the final training, the selected model showed the following F1-scores:
 
  Class high_risk: F1 = 0.927
  
  Class low_risk: F1 = 0.828
  
  Class mid_risk: F1 = 0.746
---

## 4. Project & Deployment Overview  
This repository contains:  
- A **Dockerized FastAPI** application that wraps the trained ML model. While executed, it returns the predicted risk level (high_risk, mid_risk or low_risk). It includes Dockerfile, uv.lock, and pyproject.toml.
- The **model binary** file (`MR_model.bin`) used by the application.
- The **train.py** file with training the final model.
- The **predict.py** file with loading the model via FastAPI.
- The **Notebook.ipynb** file with data preparation and cleaning, as well as with features and model selection process.
- The **test.txt** which contains pre-processed data form the dataset. To test the model, you can copy a dictionary containing health-related info of a person from *Maternal Health Risk Data*, use it as an input and recieve a prediction. 

The application was deployed via [Render](https://render.com/). It is accessible via this link:  


Be aware that the application needs time to swicth from sleeping to an active mode. Moreover, Render tends to show an error while executing the link for the first time. Please, reload the page, it you see an error or an empty scree, this should help.

---

## 5. How to Use the Model  

1. **Test File**: A sample input file `test.txt` is provided in this repo.  
   - Copy one of the feature-dictionary entries from `test.txt`.   

2. **Prediction via Web UI**:  
   - Visit the API docs: 
   - Click on **Try it out** button to execute the model
   - Replace *"{"additionalProp1": {} }* with a dictionary containing health info of one person from *test.txt*, the click on the **Execute** button.
   - The model will return:  
        **"Maternal risk**": the predicted risk level (high_risk, mid_risk or low_risk)

**For example**, if you imput is:
```
{"Age": 25,
 "SystolicBP": 120,
 "DiastolicBP": 90,
 "BS": 12.0,
 "BodyTemp": 101.0,
 "HeartRate": 80}
```

**The model will reply with:**
```
{
  "Maternal risk": "['high_risk']"
}
```
## 6. Local Development
If you want to run the FastAPI + model locally, use the following code:

```
cd Maternal-risk
docker build -t  maternal-risk .
docker run -p 10000:10000  maternal-risk
```

Then navigate to http://localhost:10000/docs to test the API locally.

## 7. References and acknowledgments 

Kaggle dataset: [Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data) 

**References**:

- Ahmed M., Kashem M.A., Rahman M., Khatun S. (2020) Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT). In: Kasruddin Nasir A. et al. (eds) InECCE2019. Lecture Notes in Electrical Engineering, vol 632. Springer, Singapore. [Web Link]
- IoT based Risk Level Prediction Model for Maternal Health Care in the Context of Bangladesh, STI-2020, [under publication in IEEE]`

I acknowledge the use of ChatGPT for debugging, search for deployment options and drafting the README file. 