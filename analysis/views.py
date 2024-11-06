from django.conf import settings
import os
import pandas as pd
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.template import context
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# add this line
from django.templatetags.static import static
from scipy import stats
from scipy.stats import ttest_ind

from .my_data import load_and_clean_data
    
import csv

# Load the CSV file (optional global loading, but generally better to do in views)

def convert_age_to_midpoint(age_range):
    age_range = age_range.strip('[]()')  # Remove any brackets or parentheses
    low, high = age_range.split('-')  # Split the range into low and high values
    return (int(low) + int(high)) / 2  # Calculate the midpoint


def data_analysis(request):
    # Load the data
    csv_file_path = os.path.join(settings.BASE_DIR, 'static', 'hospital_readmissions.csv')
    df_health = pd.read_csv(csv_file_path)

    # Call load_and_clean_data function to clean data
    df_health, duplicates, null_values = load_and_clean_data(df_health)

    # Example: Create a chart (e.g., countplot of readmissions)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='age_cat', data=df_health, hue='readmitted')  # Use countplot instead of histplot
    chart_path = os.path.join(settings.BASE_DIR, 'static', 'charts', 'age_cat_countplot.png')
    plt.savefig(chart_path)
    plt.close()
    
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='pry_diagnosis', data=df_health, hue='readmitted')  # Use countplot instead of histplot
    chart_pry_diag_path = os.path.join(settings.BASE_DIR, 'static', 'charts', 'pry_diag_countplot.png')
    plt.savefig(chart_pry_diag_path)
    plt.close()
    
    
    

    # Create a pie chart for the readmitted data
    readmitted_data = {
        'no': 0.534706,
        'yes': 0.465294
    }

    # Create a figure for the pie chart
    fig, ax = plt.subplots()
    labels = [f"{status}" for status in readmitted_data.keys()]
    sizes = list(readmitted_data.values())
    colors = ['lightblue', 'lightgreen']

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Ensure pie is drawn as a circle
    plt.title('Distribution of Diabetic Melitus patients readmitted data Categories')

    pie_chart_path = os.path.join(settings.BASE_DIR, 'static', 'charts', 'readmitted_piechart.png')
    plt.savefig(pie_chart_path)
    plt.close()

    # Create a pie chart for the readmitted data
    non_readmitted_data = {
        'no': 527202,
        'yes': 472798
    }

    # Create a figure for the pie chart
    fig, ax = plt.subplots()
    labels = [f"{status}" for status in non_readmitted_data.keys()]
    sizes = list(non_readmitted_data.values())
    colors = ['pink', 'red'] 

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Ensure pie is drawn as a circle
    plt.title('Distribution of Non-Diabetic Melitus patients readmitted data Categories') 

    pie_chart_non_diab_path = os.path.join(settings.BASE_DIR, 'static', 'charts', 'non_readmitted_piechart.png') 
    plt.savefig(pie_chart_non_diab_path)
    plt.close()
    
    
    
    
    

    # Add pie chart URL to context
    # Apply the function to convert age ranges to midpoints
    df_health['age_cat_mid'] = df_health['age'].apply(convert_age_to_midpoint)

    # Convert 'readmitted' column to numerical (0 and 1)
    df_health['readmitted'] = np.where(df_health['readmitted'] == 'yes', 1, 0)
    
    # Heatmap for correlation among numerical variables
    numerical_df = df_health.select_dtypes(include=['float64', 'int64'])  # Select numerical columns
    plt.figure(figsize=(20, 16))
    sns.heatmap(numerical_df.corr(), cmap='YlGnBu', annot=True, fmt='.2g')
    plt.title('Heatmap showing correlation among numerical variables', y=1.03)
    heatmap_path = os.path.join(settings.BASE_DIR, 'static', 'charts', 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    
    
    # Descriptive statistics
    descriptive_stats = df_health.describe()
    categorical_stats = df_health.describe(include='category')

    # Variance calculation for numerical columns
    numerical_columns = df_health.select_dtypes(include=['float64', 'int64']).drop(columns=['readmitted'], errors='ignore')
    numerical_variance = numerical_columns.var()

    # T-test for age and readmission
    readmitted = df_health[df_health['readmitted'] == 1]['age_cat_mid']
    not_readmitted = df_health[df_health['readmitted'] == 0]['age_cat_mid']
    t_stat, p_val = ttest_ind(readmitted, not_readmitted)
    mean_readmitted = readmitted.mean()
    mean_not_readmitted = not_readmitted.mean()

    
    

    # Pass data and charts to the template
    context = {
        'data': df_health.head().to_dict(orient='records'),
        'duplicates': duplicates,
        'null_values': null_values,
        'descriptive_stats': descriptive_stats.to_html(),
        'categorical_stats': categorical_stats.to_html(),
        'numerical_variance': numerical_variance.to_dict(),
        't_stat': t_stat,
        'p_val': p_val,
        'mean_readmitted': mean_readmitted,
        'mean_not_readmitted': mean_not_readmitted,
        'chart_url': '/static/charts/age_cat_countplot.png',  # Update chart URL
        'chart_pry_diag_url': '/static/charts/pry_diag_countplot.png',
        'pie_chart_url': '/static/charts/readmitted_piechart.png',  # Update chart URL
        'pie_chart_non_diab_url' : '/static/charts/non_readmitted_piechart.png',
        'age_chart': '/static/charts/age_distribution.png',  # relative path for use in the template
        'heatmap_url': '/static/charts/correlation_heatmap.png'  # Add heatmap to context
    }

    return render(request, 'data_analysis.html', context)

    
