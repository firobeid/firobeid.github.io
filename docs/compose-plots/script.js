importScripts("https://cdn.jsdelivr.net/pyodide/v0.21.3/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.0/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.0/dist/wheels/panel-0.14.0-py3-none-any.whl', 'github', 'holoviews>=1.15.1', 'hvplot', 'numpy', 'pandas']
  for (const pkg of env_spec) {
    const pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    await self.pyodide.runPythonAsync(`
      import micropip
      await micropip.install('${pkg}');
    `);
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

import numpy as np
import panel as pn
from pathlib import Path
import pandas as pd
import hvplot.pandas
from io import BytesIO


'''
<meta http-equiv="pragma" content="no-cache" />
<meta http-equiv="expires" content="-1" />
'''

hospital_data = pd.read_csv(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/hospital_claims.csv'
).dropna()

# Slice the DataFrame to consist of only "552 - MEDICAL BACK PROBLEMS W/O MCC" information
procedure_552_charges = hospital_data[
    hospital_data["DRG Definition"] == "552 - MEDICAL BACK PROBLEMS W/O MCC"
]
# Group data by state and average total payments, and then sum the values
payments_by_state = procedure_552_charges[["Average Total Payments", "Provider State"]]
# Sum the average total payments by state
total_payments_by_state = payments_by_state.groupby("Provider State").sum()
plot1 = total_payments_by_state.hvplot.bar(rot = 45)


# Sort the state data values by Average Total Paymnts
sorted_total_payments_by_state = total_payments_by_state.sort_values("Average Total Payments")
sorted_total_payments_by_state.index.names = ['Provider State Sorted']
# Plot the sorted data
plot2 = sorted_total_payments_by_state.hvplot.line(rot = 45)

sorted_total_payments_by_state.index.names = ['Provider State Sorted']
plot3 = total_payments_by_state.hvplot.bar(rot = 45) + sorted_total_payments_by_state.hvplot(rot = 45)

# Group data by state and average medicare payments, and then sum the values
medicare_payment_by_state = procedure_552_charges[["Average Medicare Payments", "Provider State"]]
total_medicare_by_state = medicare_payment_by_state.groupby("Provider State").sum()
# Sort data values
sorted_total_medicare_by_state = total_medicare_by_state.sort_values("Average Medicare Payments")
plot4 = sorted_total_medicare_by_state.hvplot.bar(rot = 45)

plot5 = sorted_total_payments_by_state.hvplot.line(label="Average Total Payments", rot = 45) * sorted_total_medicare_by_state.hvplot.bar(label="Average Medicare Payments", rot = 45)

# Overlay plots of the same type using * operator
plot6 = sorted_total_payments_by_state.hvplot.bar(label="Average Total Payments", rot = 45) * sorted_total_medicare_by_state.hvplot.bar(label="Average Medicare Payments", width = 1000, rot = 45)

# hvplot_snip = pn.pane.HTML("https://firobeid.github.io/compose-plots/Resources/binning_V1.html")
hvplot_snip = pn.pane.Markdown("""[DataViz HTMLS Deployments](https://firobeid.github.io/compose-plots/Resources/binning_V1.html)""")
pn.extension( template="fast")

pn.state.template.param.update(
    # site_url="",
    # site="",
    title="UCBerkely FinTech Bootcamp Demo",
    favicon="https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/favicon.ico",
)
# Create a Title for the Dashboard
title = pn.pane.Markdown(
    """
# UCBerkley FinTech Bootcamp Demo - Firas Obeid
""",
    width=1000,
)
title1 = pn.pane.Markdown(
    """
# Hospital Data Analysis
""",
    width=800,  
)

title2 = pn.pane.Markdown(
    """
# Machine Learning Unwinding 
""",
    width=800,  
)

image = pn.pane.image.PNG(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/image.png',
    alt_text='Meme Logo',
    link_url='https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/image.png',
    width=500
)
welcome = pn.pane.Markdown(
    """
### This dashboard/WebApp leverages FinTech and Data Science tools for practical and hands on demo's for UCBerkley FinTech Bootcamp students in [\`Firas Ali Obeid's\`](https://www.linkedin.com/in/feras-obeid/) classes
* Motive is to keep students up to date with the tools that allows them to define a problem till deployment in a very short amount of time for efficient deliverables in the work place or in academia. 
* The tool/web app is developed completly using python and deployed serverless on github pages (not static anymore right?! 

* Disclaimer: All data presented are from UCBerkley resources.
* Disclaimer: All references: https://blog.holoviz.org/panel_0.14.html

***\`Practice what you preach\`***

"""
)
#ML GENERAL
ml_slider = pn.widgets.IntSlider(start=1, end=10)
def ml_slideshow(index):
    url = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/{index}.png"
    return pn.pane.JPG(url, width = 500)

ml_output = pn.bind(ml_slideshow, ml_slider)
ml_app = pn.Column(ml_slider, ml_output)

##CLUSTERING
clustering_slider = pn.widgets.IntSlider(start=1, end=36)
def cluster_slideshow(index):
    url2 = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/Clustering/Clustering-{index}.png"
    return pn.pane.PNG(url2,width = 800)
cluster_output = pn.bind(cluster_slideshow, clustering_slider)
# cluster_app = pn.Column(clustering_slider, cluster_output)
k_means_simple = pn.pane.Markdown("""
### K_means Simple Algo Implementation
\`\`\`python

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels
\`\`\`
""",width = 500)

##GENERAL ML
general_ml_slider = pn.widgets.IntSlider(start=1, end=17)
def general_ml_slideshow(index):
    url = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Algo_Survey/{index}.png"
    return pn.pane.PNG(url,width = 800)
general_ml_output = pn.bind(general_ml_slideshow, general_ml_slider)

ML_algoes = pn.pane.Markdown("""
### Some behind the Scenes Simple Implementations
\`\`\`python

import numpy as np

def LogesticRegression_predict(features, weights, intercept):
    dot_product = np.dot(features,weights.T) #or .reshape(-1) instead of T
    z = intercept + dot_product 
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

import pickle
def save_model(model_name, model):
    '''
    model_name = name.pkl
    joblib.load('name.pkl')
    assign a variable to load model
    '''
    with open(str(model_name), 'wb') as f:
        pickle.dump(model, f)
\`\`\`

### Criteria for Splitting in Decision Tress
\`\`\`python
def gini(rows):
    '''
    Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    '''
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity
\`\`\`
### Find Best Split Algo (Decision Tree)

\`\`\`python
def find_best_split(rows):
    '''Find the best question to ask by iterating over every feature / value
    and calculating the information gain.'''
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question
\`\`\`

#### Why we doing Label Encoding?
- We apply One-Hot Encoding when:

The categorical feature is not ordinal (like the countries above)
The number of categorical features is less so one-hot encoding can be effectively applied

- We apply Label Encoding when:

The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
The number of categories is quite large as one-hot encoding can lead to high memory consumption
\`\`\`python
categorical_vars = list(df.columns[df.dtypes == object].values)
obj_df = df.select_dtypes(include=['object']).copy() 
map_dict = {col: {n: cat for n, cat in enumerate(obj_df[col].astype('category').cat.categories)} for col in obj_df}
obj_df = pd.DataFrame({col: obj_df[col].astype('category').cat.codes for col in obj_df}, index=obj_df.index)

\`\`\`
""",width = 500)

ML_metrics =  pn.pane.Markdown("""
### Binary Classification Metrics Calculation

\`\`\`python
__author__: Firas Obeod
def metrics(matrix):
    '''
    Each mean is appropriate for different types of data; for example:

    * If values have the same units: Use the arithmetic mean.
    * If values have differing units: Use the geometric mean.
    * If values are rates: Use the harmonic mean.
    '''
    TN = matrix[0,0]
    FP = matrix[0,1]
    FN = matrix[1,0]
    TP = matrix[1,1]
    Specificity =  round(TN / (FP + TN), 4) # True Negative Rate 
    FPR  = round(FP / (FP + TN), 4)
    Confidence = round(1 - FPR, 4)
    FDR = round(FP / (FP + TP), 4)
    Precision = 1 - FDR # TP / (FP + TP)
    Recall_Power = round(TP / (TP + FN), 4) #Sensitivity or TPR
    G_mean = (Specificity * Recall_Power) **(1/2) 
    Accuracy = round((TP + TN) / (TP +FP + TN + FN), 4)
    return {'FPR':FPR, 'Confidence': Confidence, 'FDR' :FDR, 'Precision': 
            Precision, 'Recall_Power':Recall_Power, 'Accuracy': Accuracy, "G_mean": G_mean}
\`\`\`
""", width = 500)
##TIMESERIES
timeseries_libs = pn.pane.Markdown("""
## 10 Time-Series Python Libraries in 2022:

### 📚 Flow forecast

Flow forecast is a deep learning for time series forecasting framework. It provides the latest models (transformers, attention models, GRUs) and cutting edge concepts with interpretability metrics. It is the only true end-to-end deep learning for time series forecasting framework.

### 📚 Auto_TS

Auto_TS train multiple time series models with just one line of code and is a part of autoML.

### 📚 SKTIME

Sktime an extension to scikit-learn includes machine learning time-series for regression, prediction, and classification. This library has the most features with interfaces scikit-learn, statsmodels, TSFresh and PyOD.

### 📚 Darts

Darts contains a large number of models ranging from ARIMA to deep neural networks. It also lets users combine predictions from several models and external regressors which makes it easier to backtest models.

### 📚 Pmdarima

Pmdarima is a wrapper over ARIMA with automatic Hyperparameter tunning for analyzing, forecasting, and visualizing time series data including transformers and featurizers, including Box-Cox and Fourier transformations and a seasonal decomposition tool.

### 📚 TSFresh

TSFresh automates feature extraction and selection from time series. It has Dimensionality reduction, Outlier detection and missing values.

### 📚 Pyflux

Pyflux builds probabilistic model, very advantageous for tasks where a more complete picture of uncertainty is needed and the latent variables are treated as random variables through a joint probability.


### 📚 Prophet

Facebook's Prophet is a forecasting tool for CSV format and is suitable for strong seasonal data and robust to missing data and outliers.
Prophet is a library that makes it easy for you to fit a model that decomposes a time series model into trend, season, and holiday components. It's somewhat customizable and has a few nifty tools like graphing and well-thought out forecasting.
Prophet does the following linear decomposition:

* g(t): Logistic or linear growth trend with optional linear splines (linear in the exponent for the logistic growth). The library calls the knots 'change points.'
* s(t): Sine and cosine (i.e. Fourier series) for seasonal terms.
* h(t): Gaussian functions (bell curves) for holiday effects (instead of dummies, to make the effect smoother).

[Some thoughts about Prophet](https://www.reddit.com/r/MachineLearning/comments/syx41w/p_beware_of_false_fbprophets_introducing_the/)

### 📚 Statsforecast
[GitHub Link to Statsforecast](https://github.com/Nixtla/statsforecast)

Statsforecast offers a collection of univariate time series. It invludes ADIDA, HistoricAverage, CrostonClassic, CrostonSBA, CrostonOptimized, SeasonalNaive, IMAPA Naive, RandomWalkWithDrift, TSB, AutoARIMA and ETS.
Impressive fact: It is 20x faster than pmdarima , 500x faster than Prophet,100x faster than NeuralProphet, 4x faster than statsmodels. 

### 📚 PyCaret

PyCaret replaces hundreds of lines of code with few lines only. Its time-series forecasting is in pre-release mode with --pre tag with 30+ algorithms. It includes automated hyperparameter tuning, experiment logging and deployment on cloud.

### 📚 NeuralProphet

NeuralProphet is a Neural Network based Time-Series model, inspired by Facebook Prophet and AR-Net, built on PyTorch.

Source: Maryam Miradi, PhD 
""",width = 800)
timeseries_data_split = pn.pane.Markdown("""
### Training and Validating Time Series Forecasting Models
\`\`\`python

from sklearn.model_selection import TimeSeriesSplit
N_SPLITS = 4


X = df['timestamp']
y = df['value']


folds = TimeSeriesSplit(n_splits = N_SPLITS)


for i, (train_index, valid_index) in enumerate(folds.split(X)):
	X_train, X_valid = X[train_index], X[valid_index]
	y_train, y_valid = y[train_index], y[valid_index]
\`\`\`
### Training and Validating \`Financial\` Time Series Forecasting Models
\`\`\`python

__author__ = 'Stefan Jansen'
class MultipleTimeSeriesCV:
    '''
    Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes
    '''

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 date_idx='date',
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[(dates[self.date_idx] > days[train_start])
                              & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                             & (dates[self.date_idx] <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
\`\`\`
""",width = 800)
ts_gif = pn.pane.GIF("https://raw.githubusercontent.com/firobeid/machine-learning-for-trading/main/assets/timeseries_windowing.gif")
ts_cv = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ts/cv.png",link_url = 'https://wandb.ai/iamleonie/A-Gentle-Introduction-to-Time-Series-Analysis-Forecasting/reports/A-Gentle-Introduction-to-Time-Series-Analysis-Forecasting--VmlldzoyNjkxOTMz', width = 800)
# Create a tab layout for the dashboard
# https://USERNAME.github.io/REPO_NAME/PATH_TO_FILE.pdf
motivational = pn.pane.Alert("## YOUR PROGRESS...\\nUpward sloping and incremental. Keep moving forward!", alert_type="success")
gif_pane = pn.pane.GIF('https://upload.wikimedia.org/wikipedia/commons/b/b1/Loading_icon.gif')
progress_ = pn.pane.PNG('https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/Progress.png')

##########################
##TIMESERIES COMPETITION##
##########################
reward = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/TimeSeriesCompetition/Images/Reward.png")
other_metrics = pn.pane.PNG("https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ts/Regression_Loss_functions.png", height = 500)
def cal_error_metrics():
    global real_test_data, predictions, rmse_error

    def rmse(preds,target):
        if (len(preds)!=len(target)):
            raise AttributeError('list1 and list2 must be of same length')
        return round(((sum((preds[i]-target[i])**2 for i in range(len(preds)))/len(preds)) ** 0.5),2)

    try:
        assert len(real_test_data) == len(predictions)
    except Exception as e: # if less than 2 words, return empty result
        return pn.pane.Markdown("""ERROR:You didnt upload excatly 17519 predictions rows!!""")
    try:
        rmse_error = rmse(real_test_data["GHI"].values, predictions[predictions.columns[0]].values)

        error_df = pd.DataFrame({"RMSE":[rmse_error]}, index = [str(file_input_ts.filename)])
        error_df.index.name = 'Uploaded_Predictions'
    except Exception as e: 
        return pn.pane.Markdown(f"""{e}""")

    return pn.widgets.DataFrame(error_df, width=300, height=100, name = 'Score Board')


def get_real_test_timeseries():
    global real_test_data, predictions 
    real_test_data = hospital_data = pd.read_csv(
    'https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/TimeSeriesCompetition/test_data/competition_real_test_data_2018.csv'
).dropna()
    if file_input_ts.value is None:
        predictions = pd.DataFrame({'GHI': [real_test_data['GHI'].mean()] * len(real_test_data)})
    else:
        predictions = BytesIO()
        predictions.write(file_input_ts.value)
        predictions.seek(0)
        print(file_input_ts.filename)
        try:
            predictions = pd.read_csv(predictions, error_bad_lines=False).dropna()#.set_index("id")
        except:
            predictions = pd.read_csv(predictions, error_bad_lines=False).dropna()
        if len(predictions.columns) > 1:
            predictions = predictions[[predictions.columns[-1]]]
        predictions = predictions._get_numeric_data()
        predictions[predictions < 0] = 0 #predictions cant be hegative for solar energy prediction task
        # New_Refit_routing = New_Refit_routing[[cols for cols in New_Refit_routing.columns if New_Refit_routing[cols].nunique() >= 2]] #remove columns with less then 2 unique values
    # return predictions

def github_cred():
    from github import Github
    repo_name = 'firobeid/TimeSeriesCompetitionTracker'
    # using an access token
    g = Github("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")
    return g.get_repo(repo_name)

def leaderboard_ts():
    global file_on_github
    # repo_name = 'firobeid/TimeSeriesCompetitionTracker'
    # # using an access token
    # g = Github("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")
    # # Create Github linkage Instance
    # g = github_cred()
    # if prediction_submission_name.value == 'Firas_Prediction_v1':
    repo = github_cred()
    contents = repo.get_contents("")
    competitior_rank_file = 'leadership_board_ts.csv'
    if competitior_rank_file not in [i.path for i in contents]:
        print("Creatine leaderboard file...")
        repo.create_file(competitior_rank_file, "creating timeseries leaderboard", "Competitor_Submission, RMSE", branch="main")
    file_on_github = pd.read_csv("https://raw.githubusercontent.com/firobeid/TimeSeriesCompetitionTracker/main/leadership_board_ts.csv", delim_whitespace=" ") 

def upload_scores():
    global rmse_error, sub_name, file_on_github
    competitior_rank_file = 'leadership_board_ts.csv'
    repo = github_cred()
    submission = sub_name
    score = rmse_error
    leaderboard_ts()
    file_on_github.loc[len(file_on_github.index)] = [submission, score]

    target_content = repo.get_contents(competitior_rank_file)
    repo.update_file(competitior_rank_file, "Uploading scores for %s"%sub_name,  file_on_github.to_string(index=False), target_content.sha, branch="main")
    return pn.pane.Markdown("""Successfully Uploaded to Leaderboard!""")

def final_github():
    global sub_name
    global real_test_data, predictions, rmse_error
    sub_name = str(prediction_submission_name.value.replace("\\n", "").replace(" ", ""))
    print(sub_name)
    if 'rmse_error' not in globals(): #not to allow saving rmse everytime site is reoaded
        return pn.widgets.DataFrame(file_on_github.sort_values(by = 'RMSE',ascending=True).set_index('Competitor_Submission'), width=600, height=1000, name = 'Leader Board')
    
    else:
        try:
            if sub_name != 'Firas_Prediction_v1': #not to allow saving rmse everytime site is reoaded also
                upload_scores()
        except Exception as e: 
            return pn.pane.Markdown(f"""{e}""")
        file_on_github["Rank"] = file_on_github.rank(method = "min")["RMSE"]
        return pn.widgets.DataFrame(file_on_github.sort_values(by = 'RMSE',ascending=True).set_index('Rank'), width=600, height=1000, name = 'Leader Board')

run_github_upload = pn.widgets.Button(name="Click to Upload Results to Leaderscore Board!")
prediction_submission_name  = pn.widgets.TextAreaInput(value="Firas_Prediction_v1", height=100, name='Change the name of submission below:')
widgets_submission = pn.WidgetBox(
    pn.panel("""# Submit to LeaderBoard Ranking""", margin=(0, 10)),
    pn.panel('* Change Submision Name Below to your own version and team name (no spaces in between)', margin=(0, 10)),
    prediction_submission_name,
    run_github_upload, 
    pn.pane.Alert("""##                Leader Ranking Board""", alert_type="success",),
    width = 500
)

# def update_submission_widget(event):
#     global sub_name
#     prediction_submission_name.value = event.new
#     sub_name = str(prediction_submission_name.value.replace("\\n", "").replace(" ", ""))
#     print(sub_name)
# # when prediction_submission_name changes, 
# # run this function to global variable sub_name
# prediction_submission_name.param.watch(update_submission_widget, "value")

@pn.depends(run_github_upload.param.clicks)
def ts_competition_submission(_):
    leaderboard_ts()
    return pn.Column(final_github)


run_button = pn.widgets.Button(name="Click to get model error/score!")
file_input_ts = pn.widgets.FileInput(align='center')
text_ts = """
# Prediction Error Scoring

This section is to host a time series modelling competition between UCBekely students teams'. The teams should
build a time series univariate or multivariate model but the aim is to forcast the \`GHI\` column (a solar energy storage metric).

The train data is 30 minutes frequecy data between 2010-2017 for solar energy for UTDallas area. The students then predict the whole off 2018
,which is 17519 data points (periods) into the future (2018). The students submit there predictions as csv over here, 
get error score (RMSE not the best maybe but serves learning objective) and submit to leaderboard to be ranked. Public submissions
are welcome! But I cant give you extra points on project 2 ;)

The data used for the modelling can be found here: 
[Competition Data](https://github.com/firobeid/Forecasting-techniques/tree/master/train_data)

### Instructions
1. Upload predictions CSV (only numerical data)
2. Make sure you have 17519 predictions / row in your CSV and only one column
3. Press \`Click to get model error/score!\`
4. Observe you predictions error under yellow box bellow
5. If satisfied move on to the next box to the right to submit team name and prediction. 
\`My code takes care of pulling your error and storing it on GitHub to be ranked against incoming scores from teams\`
"""
widgets_ts = pn.WidgetBox(
    pn.panel(text_ts, margin=(0, 10)),
    pn.panel('Upload Prediction CSV', margin=(0, 10)),
    file_input_ts,
    run_button, 
    pn.pane.Alert("### Prediction Results Will Refresh Below After Clicking above", alert_type="warning")
    , width = 500
)

def update_target(event):
    get_real_test_timeseries()

file_input_ts.param.watch(update_target, 'value')

@pn.depends(run_button.param.clicks)
def ts_competition(_):
    get_real_test_timeseries()
    return pn.Column(cal_error_metrics)


#########
##FINAL##
#########
tabs = pn.Tabs(
    ("Welcome", pn.Column(welcome, image)
    ),
    ("DataViz",pn.Tabs(("Title",pn.Column(pn.Row(title1),hvplot_snip)),
                    ("total_payments_by_state", pn.Row(plot1)),
                    ("sorted_total_payments_by_state", pn.Row(plot2)),
                    ("Tab1 + Tab2", pn.Column(plot3,width=960)),
                    ("sorted_total_medicare_by_state", pn.Row(plot4,plot5, plot6, width=2000))
                      )
    ),
    ("Zen of ML", pn.Tabs(("Title",pn.Row(title2,gif_pane, pn.Column(motivational,progress_))),
                          ('Lets Get Things Straight',pn.Column(ml_slider, ml_output)),
                          ('Unsupervised Learning (Clustering)', pn.Row(pn.Column(clustering_slider, cluster_output),k_means_simple)),
                          ("TimeSeries Forecasting",pn.Row(timeseries_libs,pn.Column(ts_gif, ts_cv),timeseries_data_split)),
                          ("General ML Algorithms' Survey", pn.Row(pn.Column(general_ml_slider, general_ml_output),ML_algoes, ML_metrics)),
                          ('TimeSeries Competition Error Metric',pn.Row(pn.Column(widgets_ts, ts_competition, reward), pn.layout.Spacer(width=20), pn.layout.Spacer(width=20), pn.Column(pn.pane.Markdown("### Other Metrics Can Be Used:"),other_metrics))) 
                          #('TimeSeries Competition Error Metric',pn.Row(pn.Column(widgets_ts, ts_competition, reward), pn.layout.Spacer(width=20), pn.Column(widgets_submission, ts_competition_submission), pn.layout.Spacer(width=20), pn.Column(pn.pane.Markdown("### Other Metrics Can Be Used:"),other_metrics))) 
                         )
    )
    )

audio = pn.pane.Audio('http://ccrma.stanford.edu/~jos/mp3/pno-cs.mp3', name='Audio')
pn.Column(pn.Row(title), tabs, pn.Row(pn.pane.Alert("Enjoy some background classic", alert_type="success"),audio), ).servable(target='main')


await write_doc()
  `
  const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
  self.postMessage({
    type: 'render',
    docs_json: docs_json,
    render_items: render_items,
    root_ids: root_ids
  });
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()