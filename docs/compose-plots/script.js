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
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.0/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.0/dist/wheels/panel-0.14.0-py3-none-any.whl', 'holoviews>=1.15.1', 'hvplot', 'numpy', 'pandas']
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
### This dashboard presents a visual analysis of hospital data for a demo to UCBerkley FinTech Bootcamp students in [\`Firas Obeid's\`](https://www.linkedin.com/in/feras-obeid/) classes
* Motive is to keep students up to date with the tools that allows them to define a problem till deployment in a very short amount of time for efficient deliverables in the work place or in academia.
* Disclaimer: All data presented are from UCBerkley resources.
* Disclaimer: All references: https://blog.holoviz.org/panel_0.14.html

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
general_ml_slider = pn.widgets.IntSlider(start=1, end=11)
def general_ml_slideshow(index):
    url = f"https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/ML_lectures/ML_Algo_Survey/{index}.png"
    return pn.pane.PNG(url,width = 800)
general_ml_output = pn.bind(general_ml_slideshow, general_ml_slider)



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
                          ("General ML Algorithms' Survey", pn.Column(general_ml_slider, general_ml_output))
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