# Parkinson’s Disease Detection using Machine Learning

## 1. Problem Statement  
Parkinson’s disease (PD) is a neurodegenerative disorder that affects millions of people worldwide. Early detection is crucial for improving patient outcomes, as treatments and interventions can be more effective when initiated sooner. Traditional diagnostic procedures may be time-consuming, costly, or subjective. By leveraging a machine learning model, I aim to build an **automatic prediction system** that estimates the probability a patient has Parkinson’s disease — providing a fast, objective, and scalable screening tool.

Using ML for this problem is particularly powerful because:
- It can combine multiple clinical and biometric features to detect subtle patterns.  
- Once deployed, it can run in a low-cost, automated pipeline — useful for remote or resource-limited settings.  
- A probabilistic model (rather than a simple “yes/no” classifier) provides risk scores, which clinicians can use to inform further testing.

---

## 2. Dataset Description  
For my project, I used the *Parkinson’s Disease Dataset Analysis* dataset from Kaggle. The dataset contains comprehensive health information for 2105 people, from which 1304 individuals were diagnosed with Parkinsons' disease (PD).   

- **Source**: [Kaggle Parkinson’s Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/parkinsons-disease-dataset-analysis)  
- **DOI**: `https://doi.org/10.34740/kaggle/dsv/8668551`

The dataset includes features related to voice measurements, motor function, and clinical assessments, which have been shown in literature to correlate with Parkinson’s progression.

The dataset can be downloaded direclty form Kaggle following the link above manually or automatically as shown in my Notebook.ipynb. 

---

## 3. Feature Selection & Model Choice  

The features and model selection process is given step-by-step in Notebook.ipynb. To execute this file, download it to your device. You do not need to download the dataset, the code will do this automatically. You only need to ensure that all the packages listed in Requirments are installed.

- **Feature Selection**:  
  - First, I explored feature importance via risk ratio and mutual information metrics (categorical variables), and via correlation analysis (numerical variables) to narrow down the most predictive variables. Overall, I have chosen 12 predictors.
  - Second, I buit two logistic regression models to test which set of parametres is better (restricted or the full). Based on the AUC score, I've chose the restricted set with 12 the most predictive variables for my futher models.

- **Model Choice**:  
  - I trained and compared multiple models: **Logistic Regression**, **Decision Trees**, **Random Forests**, and **Gradient Boosting**.  
  - The final model was selected based on its predictive performance, measures by ROC-AUC and F1-score, after the parameter tunning procedures. **RandomForestClassifier** showed the best perdormance (AUC score = 0.969984), overperformed even the Gradient Boosting model.
  - For the selected model, I found the best threshold *0.45*, based on F1-score.
  - After the final training, the selected model showed the accuracy of 0.912 and AUC-score = 0.961 and F1-score = 0.93.

---

## 4. Project & Deployment Overview  
This repository contains:  
- A **Dockerized FastAPI** application that wraps the trained ML model. While executed, it returns the model's probability and binary prediction for a specific person to have Parkinson's disease. It includes Dockerfile, uv.lock, and pyproject.toml.
- The **model binary** file (`PD_model.bin`) used by the application.
- The **train.py** file with training the final model.
- The **predict.py** file with loading the model via FastAPI.
- The **Notebook.ipynb** file with data preparation and cleaning, as well as with features and model selection process.
- The **test.txt** which contains pre-processed data form the dataset. To test the model, you can copy a dictionary containing health-related info of a person from *Parkinson's desiase dataset*, use it as an input and recieve a prediction. 

The application was deployed via [Render](https://render.com/). It is accessible via this link:  
[**Open the live model →**](https://pd-detection.onrender.com/docs)

Be aware that the application needs time to swicth from sleeping to an active mode. Moreover, Render tends to show an error while executing the link for the first time. Please, reload the page, it you see an error or an empty scree, this should help.

---

## 5. How to Use the Model  

1. **Test File**: A sample input file `test.txt` is provided in this repo.  
   - Copy one of the feature-dictionary entries from `test.txt`.   

2. **Prediction via Web UI**:  
   - Visit the API docs: https://pd-detection.onrender.com/docs
   - Click on **Try it out** button to execute the model
   - Replace *"{"additionalProp1": {} }* with a dictionary containing health info of one person from *test.txt*, the click on the **Execute** button.
   - The model will return:  
     - **Probability**: the likelihood of having Parkinson’s disease  
     - **Prediction**: `true` or `false` (based on the decision threshold)

**For example**, if you imput is:
```
{   "Tremor": "no",

    "Rigidity": "no",

    "Bradykinesia": "no",

    "PosturalInstability": "no",

    "UPDRS": 181.6204672161393,

    "FunctionalAssessment": 0.1671439736383717,

    "MoCA": 25.169071076613093,

    "Age": 63,

    "SleepQuality": 7.062629501539838,

    "BMI": 16.411189332470087,

    "AlcoholConsumption": 15.186769323417664,

    "DietQuality": 8.280780531938863 }
```

**The model will reply with:**

"Probability of being diagnosed with PD": 0.8531189083820663,

"PD diagnosis": true

## 6. Local Development
If you want to run the FastAPI + model locally, use the following code:

```
cd Parkinsons_detect
docker build -t  pd-detect .
docker run -p 10000:10000  pd-detect
```

Then navigate to http://localhost:10000/docs to test the API locally.

## 7. References and acknowledgments 

Kaggle dataset: Parkinson’s Disease Dataset Analysis

DOI: https://doi.org/10.34740/kaggle/dsv/8668551

I acknowledge the use of ChatGPT for debugging, search for deployment options and drafting the README file. 



