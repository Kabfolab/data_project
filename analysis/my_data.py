import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from django.conf import settings
from scipy import stats
from scipy.stats import ttest_ind
#import modules and packages

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score , recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Data Loading and Cleaning Function
def load_and_clean_data(df_health):
    # Count duplicates
    duplicates = df_health.duplicated().sum()

    # Count null values
    null_values = df_health.isnull().sum().sum()

    # Perform any cleaning operations on the dataframe
    df_health = df_health.drop_duplicates()  # Example cleaning step
    
    # Creating age category column
    df_health['age_cat'] = df_health['age'].replace({
        '[70-80)': 'senior-old age', '[50-60)': 'late-middle age',
        '[60-70)': 'mid-old age', '[40-50)': 'early-middle age',
        '[80-90)': 'very senior-old', '[90-100)': 'centenarians'
    })

    
    # Converting 'age_cat' to categorical data type
    df_health['age_cat'] = df_health['age_cat'].astype('category')
    
    # Renaming columns
    df_health.rename(columns={
        'diag_1': 'pry_diagnosis', 'diag_2': 'sec_diagnosis',
        'diag_3': 'other_sec_diag', 'change': 'change_in_med',
        'A1Ctest': 'HbaTest'
    }, inplace=True)
    
    # Converting specific columns to categorical data types
    cols_to_convert = ['medical_specialty', 'pry_diagnosis', 'sec_diagnosis', 
                       'other_sec_diag', 'glucose_test', 'HbaTest', 
                       'change_in_med', 'diabetes_med', 'readmitted']
    df_health[cols_to_convert] = df_health[cols_to_convert].astype('category')
    
    
    
    # Return cleaned data, duplicates count, and null values count
    return df_health, duplicates, null_values

def plot_cat_count(df_health):
    columns = ['age_cat', 'medical_specialty', 'pry_diagnosis', 'sec_diagnosis', 
               'other_sec_diag', 'glucose_test', 'HbaTest', 'change_in_med', 
               'diabetes_med', 'readmitted']
    
    sns.set(style='darkgrid', font_scale=1.25)
    sns.set_palette('husl', 3)
    
    for c in columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=c, data=df_health, hue='readmitted')
        plt.title(f'Count Plot of {c}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save the figure, but return nothing
        plt.close()  # Close the plot to free memory# 3. Groupby and Value Counts Insights


def groupby_pry_diagnosis(df_health):
    df = df_health.groupby('age_cat')['pry_diagnosis'].value_counts(normalize=True, sort=True).unstack()
    return df

# 4. Countplot for Primary Diagnosis per Age Category
def plot_diagnosis_by_age_cat(df_health):
    cp = sns.catplot(x='pry_diagnosis', col='age_cat', kind='count', data=df_health, col_wrap=2)
    cp.set_xticklabels(rotation=90)
    plt.show()

# 5. Pie Charts for Health Condition Distribution
def plot_health_condition_distribution(df_health):
    cat_vars = ['pry_diagnosis', 'sec_diagnosis', 'other_sec_diag']
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(75, 70))

    for i, var in enumerate(cat_vars):
        if i < len(axs.flat):
            cat_counts = df_health[var].value_counts()
            axs.flat[i].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
            axs.flat[i].set_title(f'{var} Distribution')

    fig.tight_layout()
    plt.show()

# 6. Subset Data for Diabetes and Non-Diabetes Patients
def subset_diabetes_data(df_health):
    diabetes_patients = df_health[(df_health['pry_diagnosis'] == 'Diabetes') | 
                                  (df_health['sec_diagnosis'] == 'Diabetes') | 
                                  (df_health['other_sec_diag'] == 'Diabetes')]
    print('Diabetic Patients Head:')
    print(diabetes_patients.head())

    non_diabetes_patients = df_health[~((df_health['pry_diagnosis'] == 'Diabetes') | 
                                        (df_health['sec_diagnosis'] == 'Diabetes') | 
                                        (df_health['other_sec_diag'] == 'Diabetes'))]
    print('Non-Diabetic Patients Head:')
    print(non_diabetes_patients.head())
    
    print('Number of Diabetic Melitus Patients:', len(diabetes_patients.index))
    print('Number of Non-Diabetic Melitus Patients:', len(non_diabetes_patients.index))

# 7. Pie Charts for Readmitted Patients (Diabetes & Non-Diabetes)
def plot_readmission_pie_chart(diabetes_readmitted_rate, non_diabetes_readmitted_rate):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    diabetes_data = diabetes_readmitted_rate
    non_diabetes_data = non_diabetes_readmitted_rate

    axs[0].pie(diabetes_data.values(), labels=diabetes_data.keys(), autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen'])
    axs[0].axis('equal')
    axs[0].set_title('Diabetic Patients Readmission Rates')

    axs[1].pie(non_diabetes_data.values(), labels=non_diabetes_data.keys(), autopct='%1.1f%%', startangle=140, colors=['pink', 'red'])
    axs[1].axis('equal')
    axs[1].set_title('Non-Diabetic Patients Readmission Rates')

    plt.show()

# 8. Plot Numerical Distributions by Readmitted Status
def plot_numerical_distributions(df_health):
    columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 
               'n_outpatient', 'n_inpatient', 'n_emergency']
    sns.set(style='darkgrid', font_scale=1.25)
    sns.set_palette('husl', 3)

    for c in columns:
        sns.displot(x=c, data=df_health, col='readmitted', col_wrap=2)
        plt.show()

# 9. Heatmap for Correlation
def plot_correlation_heatmap(df_health):
    numerical_df = df_health.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(20, 16))
    sns.heatmap(numerical_df.corr(), cmap='YlGnBu', annot=True, fmt='.2g')
    plt.title('Heatmap Showing Correlation Among Numerical Variables')
    plt.show()

# 10. Decision Tree Classifier
def decision_tree_classifier(X_train, X_test, y_train, y_test, feature_names):
    dtc = DecisionTreeClassifier(random_state=2, max_depth=3, min_samples_leaf=0.20)
    dtc.fit(X_train, y_train)
    
    test_predictions = dtc.predict(X_test)
    train_predictions = dtc.predict(X_train)
    
    acc_train = accuracy_score(y_train, train_predictions)
    acc_test = accuracy_score(y_test, test_predictions)
    prec_score = precision_score(y_test, test_predictions)
    recall_score = recall_score(y_test, test_predictions)

    print(f"Training Accuracy: {acc_train}")
    print(f"Test Accuracy: {acc_test}")
    print(f"Precision: {prec_score}")
    print(f"Recall: {recall_score}")

    feature_importances = dtc.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    sns.barplot(data=feature_importance_df.head(), x='Importance', y='Feature')
    plt.title('Feature Importance from Decision Tree')
    plt.show()
    
    # 11. Random Forest Classifier
def random_forest_classifier(X_train, X_test, y_train, y_test, feature_names):
    rfc = RandomForestClassifier(random_state=23,n_estimators=100,max_depth=6)
    # fit the model
    rfc.fit(X_train, y_train)
    test_predictions = rfc.predict(X_test)
    train_prediction = rfc.predict(X_train)
    #check model accuracy, precison and recall score
    acc_scoreX = accuracy_score(y_train, train_prediction)
    print('Training_accuracy_score :', acc_scoreX)
    acc_score = accuracy_score(y_test, test_predictions)
    print('Testing_accuracy_score :', acc_score)
    pre_score = precision_score(y_test, test_predictions)
    print('precision_score :', pre_score)
    rec_score = recall_score(y_test, test_predictions)
    print('recall_score :',rec_score)
    
    importance = rfc.feature_importances_

    imp_df = pd.DataFrame({'feature_name':feature_names, 'Importance':importance})
    fi = imp_df.sort_values(by='Importance', ascending=False).head()

    # plot feature importance
    plt.figure(figsize=(8,6))
    sns.barplot(data=fi, x= 'Importance', y='feature_name')
    plt.xlabel('Importance', fontsize=16)
    plt.ylabel('feature_name', fontsize=16)
    plt.title('Bar Chart of Random Forest classifier Feature Importance Scores')
    plt.show()

    
    

