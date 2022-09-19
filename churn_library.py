# library doc string


# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import os
import logging
from PIL import Image, ImageDraw
os.environ['QT_QPA_PLATFORM']='offscreen'

# setting up logging environment
logging.basicConfig(
    filename=r'./logs/test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        return df
    except FileNotFoundError:
        logging.error('ERROR: File not found, please check the path.')

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # checking the shape
    df.shape
    
    # checking for nulls
    df.isnull().sum()

    # getting a description of the df
    df.describe()
    
    # setting attribution flag
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    # plotting churn hist and saving to eda folder
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    plt.savefig(r'./images/eda/churn_distribution.png')
    
    # plotting age hist and saving to eda folder
    plt.figure(figsize=(20,10))
    df['Customer_Age'].hist();
    plt.savefig(r'./images/eda/customer_age_distribution.png')
    
    # plotting marital status and saving to eda folder
    plt.figure(figsize=(20,10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(r'./images/eda/marital_status_distribution.png')
    
    # plotting total transactions and saving to eda folder
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
    plt.savefig(r'./images/eda/total_transaction_distribution.png')
    
    # plotting heatmap and saving to eda folder
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(r'./images/eda/heatmap.png')
    

def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    # setup x
    X = pd.DataFrame()
    
    # encoded column
    for col in category_lst:
        lst = []
        groups = df.groupby(col).mean()['Churn']

        for val in df[col]:
            lst.append(groups.loc[val])
        
        col_name = col + '_Churn'
        df[col_name] = lst
    
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    
    return X

def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    category_lst = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'          
    ]
    X_df = encoder_helper(df, category_lst)
    X_train, X_test, y_train, y_test = train_test_split(X_df, df, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test, X_df

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    out_img = Image.new('RGB', (500,500))
    draw = ImageDraw.Draw(out_img)
    draw.text((10,10), 'Random Forest Results')
    draw.text((10,30), 'Test Results')
    draw.text((10,50), classification_report(y_test, y_test_preds_rf))
    draw.text((10,70), 'Train Results')
    draw.text((10,90), classification_report(y_train, y_train_preds_rf))
    draw.text((10,110), '---------------------------------------------')
    draw.text((10,130), 'Logistics Regression Results')
    draw.text((10,150), 'Test Results')
    draw.text((10,170), classification_report(y_test, y_test_preds_lr))
    draw.text((10,190), 'Train Results')
    draw.text((10,210), classification_report(y_train, y_train_preds_lr))
    out_img.save(r'/images/results/classification_report.png')
    
def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(20,5))
    
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # saving plot
    plt.savefig(output_pth)

def train_models(X_train, X_test, y_train, y_test, X_df):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    
    lrc.fit(X_train, y_train)
    
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # producing classificaton report image
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    # lrc plot
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show()
    plt.savefig(r'./images/results/logistic_results.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, r'./models/rfc_model.pkl')
    joblib.dump(lrc, r'./models/logistic_model.pkl')

    # loading models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # plotting models
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show()
    plt.savefig(r'./images/results/lrc_plot.png')

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(r'./images/results/shap_values.png')

    # perform feature importances plot
    feature_importance_plot(
        model=cv_rfc.best_estimator_.feature_importances_,
        X_data=X_df,
        output_pth=r'./images/results/feature_importances.png'
    )

if __name__ == '__main__':
    # calling import data
    bank_df = import_data(r'./data/bank_data.csv')

    # calling perform eda
    perform_eda(bank_df)

    # calling perform feature engineering
    X_train, X_test, y_train, y_test, X_df = perform_feature_engineering(bank_df['Churn'])

    # calling train models
    train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        X_df
    )
