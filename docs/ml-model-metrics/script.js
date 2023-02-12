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
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.0/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.0/dist/wheels/panel-0.14.0-py3-none-any.whl', 'colorama', 'dateutil', 'holoviews>=1.15.1', 'holoviews>=1.15.1', 'hvplot', 'numpy', 'pandas', 'scipy', 'scikit-learn']
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

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import io
import sys
import os
import pandas as pd
import datetime
import gc #garabage collector
from io import BytesIO
import panel as pn
import holoviews as hv
import hvplot.pandas
from warnings import filterwarnings
'''
development env: panel serve script.py --autoreload
prod prep: panel convert script.py --to pyodide-worker --out pyodide
'''

filterwarnings("ignore")
hv.extension('bokeh')
pn.extension( "plotly", template="fast")

pn.state.template.param.update(
    # site_url="",
    # site="",
    title="Classification Model Metrics",
    favicon="https://raw.githubusercontent.com/firobeid/firobeid.github.io/main/docs/compose-plots/Resources/favicon.ico",
)
#######################
###UTILITY FUNCTIONS###
#######################
def percentage(df):
    def segment(df):
        return round(df["Count"]/df["Count"].sum(),4)
    df["percent"] = segment(df)
    return df

def AUC(group):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(group['TARGET'],group['SCORE'])
    # N = sum(group["N"])
    N = round(len(group.loc[group["TARGET"].notna()]),0)
    cols = ["AUC","Count"]
    # return trapezoidal_rule(FPR.to_numpy(),TPR.to_numpy())
    return pd.Series([auc, N], index = cols)

def ROC(group):
    from sklearn.metrics import roc_curve
    FPR,TPR,T = roc_curve(group['TARGET'],group['SCORE'])
    cols = ['TPR', 'FPR']
    return pd.concat([pd.Series(TPR),pd.Series(FPR)], keys = cols, axis = 1)

def ks(group):
    from scipy.stats import ks_2samp
    y_real = group['TARGET']
    y_proba = group['SCORE']
    
    df = pd.DataFrame()
    df['real'] = y_real
    df['proba'] = y_proba
    
    # Recover each class
    class0 = df[df['real'] == 0]
    class1 = df[df['real'] == 1]
    
    ks_ = ks_2samp(class0['proba'], class1['proba'])
    
    N = round(len(group.loc[group["TARGET"].notna()]),0)
    cols = ["KS","Count"]
    
    return pd.Series([ks_[0], N], index = cols)

def psi(df):
    '''
    https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html#:~:text=To%20calculate%20the%20PSI%20we,the%20percents%20in%20each%20bucket.
    '''
    df[df == 0] = 0.001
    sub = df.copy()
    sub = sub.iloc[:,:-1].sub(df.validation,axis = 0)
    div = df.copy()
    div= div.iloc[:,:-1].div(df.validation, axis=0)
    div = np.log(div)
    return (sub*div).sum(axis = 0)

def add_extremes_OOT(df, name:str, score:str):
    '''
    Mitigate bias in OOT/Serving/baseline set that might not have high confidence scores or low confidence scores
    :param: name: str, name of the appid column
    :param: score: str, name of the score column
    '''
    # df.loc[len(df.index)] = [np.nan, "Extreme_Case_Max", np.nan, np.nan, np.nan,994.0,0.0009,np.nan,np.nan,np.nan,np.nan]
    # df.loc[len(df.index)] = [np.nan, "Extreme_Case_Min", np.nan, np.nan, np.nan,158.0,0.9999,np.nan,np.nan,np.nan,np.nan]
    df.loc[len(df.index)] = [np.nan for i in range(0,df.shape[1])]
    df.loc[(len(df.index)-1), [name, score]] = ["Extreme_Case_Max", 0.0009]
    df.loc[len(df.index)] = [np.nan for i in range(0,df.shape[1])]
    df.loc[(len(df.index)-1), [name, score]] = ["Extreme_Case_Min", 0.9999]
    return df

def last_3months(df):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    from pandas.tseries.offsets import MonthEnd

    end_of_month = ((pd.Timestamp(datetime.now().strftime('%Y-%m-%d')) - pd.Timedelta(70, unit='D')) + relativedelta(months=-1)) + MonthEnd(0)
    start_of_month = end_of_month + MonthEnd(-3) + relativedelta(days=1)
    end_of_month = end_of_month +relativedelta(hours=23, minutes=59, seconds=59)
    print('Start Month %r --- End Month %r' % (start_of_month, end_of_month))
    try:
        date_column = list(filter(lambda x:x.endswith("DATE"),gains_df.columns))[0]
    except:
        date_column = 'CREATED_DATE'
    return df[df[date_column].between(start_of_month, end_of_month)]

def gains_table_proba(data=None,target=None, prob=None):
    data = data.copy()
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['count'] = grouped.count()['target0']
    kstable['cum_total']=(kstable['count'] / kstable['count'].sum()).cumsum()
    kstable['events']  = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable['interval_rate'] = kstable['events'] / kstable['count']
    kstable = kstable.sort_values(by="min_prob", ascending=0).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['mid_point'] = np.nan
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 4) * 100

    #Formating
    kstable["cum_total"] = kstable["cum_total"].sort_values().values
    kstable = kstable.rename(columns={"min_prob":"low", "max_prob":"high"})
    kstable['mid_point'] = round((kstable['high'] + kstable['low']) / 2, 4)
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 15)
    # print(kstable)
    #Display KS
    from colorama import Fore
    ks_3mnths = "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0]))
    print("KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    kstable['cum_eventrate']= kstable['cum_eventrate'].str.replace("%","").astype(float)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].str.replace("%","").astype(float)
    kstable.index = list(range(10,0,-1))
    kstable = kstable.iloc[::-1]
    return(kstable, ks_3mnths)

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    # https://www.kaggle.com/code/podsyp/population-stability-index
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

    return round(10 **((158.313177 - UW5_Score) /274.360149), 18)

def save_csv(df, metric):
    from io import StringIO
    sio = StringIO()
    df.to_csv(sio)
    sio.seek(0)
    return pn.widgets.FileDownload(sio, embed=True, filename='%s.csv'%metric)

###############################
###END OFF UTILITY FUNCTIONS###
###############################

text = """
#  Classification Model Metrics
## AUTHOR: [\`FIRAS ALI OBEID\`](https://www.linkedin.com/in/feras-obeid/) 
###  GNU General Public License v3.0 (GPL-3.0)
#### Developed while working at [OppFi Inc.](https://www.oppfi.com/)

This tool performs feature binning by equal intervals and by equal pouplations in each interval vs bad rate/target binary variable
To get the feature deep dive feature distribution:

1. Upload a CSV (only numerical data)

2. Choose & press on the binary (0 / 1) target column in the \`Select Target Variable\` section below

3. Press Run Analysis

4. Wait few seconds and analyze the updated charts
"""



# date = str(input('What is the name off the date column: ').upper())
# id_ = str(input('What is the name off the APP name/ID column: ').upper())
# score = str(input('What is the name off the score column (i.e UW5,DM_QL...): ').upper())
# target = str(input('What is the name off the Target column (i.e Real target values such as PD70_RATIO...: ').upper())

file_input = pn.widgets.FileInput(align='center')
date_selector = pn.widgets.Select(name='Select Date Column',)
check_date = pn.widgets.Checkbox(name = 'Check if your data has a date column \\n (otherwise keep it empty)') # T/F
target_selector = pn.widgets.Select(name='Select Target Variable(True Label)')
score_selector = pn.widgets.Select(name='Select Predictions Column(Raw Probaility)')
period_metrics = pn.widgets.Select(name='Select Period', options = ['MONTHLY','WEEKLY', 'QUARTERLY'])

date_range_ = pn.widgets.DateRangeSlider(name='Baseline Period',) #value=(start, end), start=start, end=end

random_seed = pn.widgets.IntSlider(name='Random Seed for Random Generated Data (OnSet)', value=42, start=0, end=1000, step=1)

button = pn.widgets.Button(name='Get Metrics')
widgets = pn.WidgetBox(
    pn.panel(text, margin=(0, 50)),
    pn.panel('Upload a CSV containing (Date) Highly Recommended but **optional** (Score) Probability Predictions and (y) Binary Target(True Label):', margin=(0, 10)),
    file_input,
    pn.panel('Check if your data has a date column \\n (otherwise keep it empty)'),
    check_date,
    random_seed,
    pn.panel('\\n'),
    date_selector,
    target_selector,
    score_selector,
    period_metrics,
    date_range_,
    button
)

# start, end = stocks.index.min(), stocks.index.max()
# year = pn.widgets.DateRangeSlider(name='Year', value=(start, end), start=start, end=end)
# ,id_:'ID',


def get_data():
    global df
    if file_input.value is None:
        np.random.seed(random_seed.value)
        df = pd.DataFrame({'DATE': pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = 9999)), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H"),
                        'ID': [i for i in range(10000)],
                        'SCORE':np.random.uniform(size = 10000),
                        'TARGET': np.random.choice([0,1],10000, p=[0.9,0.1])})
        # df.to_csv("test_upload.csv")
    else:
        df = BytesIO()
        df.write(file_input.value)
        df.seek(0)
        try:
                df = pd.read_csv(df, error_bad_lines=False).apply(pd.to_numeric, errors='ignore')
        except:
                df = pd.read_csv(df, error_bad_lines=False)

        df = df.select_dtypes(exclude=["category"])
        df = df.replace([np.inf, -np.inf], np.nan)
        df.columns = [i.upper() for i in df.columns]
    return df

def update_target(event):
    df = get_data()
    cols = list(df.columns)
    date_selector.set_param(options=cols)
    target_selector.set_param(options=cols)
    score_selector.set_param(options=cols)

    if check_date.value == True:
        date_column = [i.find("DATE") for i in df.columns] 
        date_column = [date_column.index(i) for i in [i for i in date_column if i !=-1]]
        if len(date_column) == 0:
            df = df.iloc[:,date_column].iloc[:,[0]]
            df.columns = ['DATE']
            start, end = df.DATE.min(), df.DATE.max()
            date_range_.set_param(value=(start, end), start=start, end=end)
        else:
            print('Creating synthetic dates')
            synthetic_date = pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = len(df) - 1)), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H")
            df['DATE'] = synthetic_date[:len(df)]
            start, end = df.DATE.min(), df.DATE.max()
            date_range_.set_param(value=(start, end), start=start, end=end)
    else:
        print('Creating synthetic dates')
        synthetic_date = pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = len(df) - 1)), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H")
        df['DATE'] = synthetic_date[:len(df)]
        start, end = df.DATE.min(), df.DATE.max()
        date_range_.set_param(value=(start, end), start=start, end=end)
file_input.param.watch(update_target, 'value')
update_target(None)

@pn.depends(button.param.clicks)
def run(_):
    print(random_seed.value)
    print(score_selector.value)
    df = get_data()
    try:
        if file_input.value is None:
            pass
        elif check_date.value == True:
            df = df.rename(columns={date_selector.value:'DATE',score_selector.value:'SCORE',target_selector.value:'TARGET'})
        else:
            synthetic_date = pd.date_range(start = (datetime.datetime.today() - pd.DateOffset(hours = len(df) - 1)), end = datetime.datetime.today(), tz = "US/Eastern", freq = "H")
            df['DATE'] = synthetic_date[:len(df)]
            df = df.rename(columns={score_selector.value:'SCORE',target_selector.value:'TARGET'})
    except Exception as e:
        return pn.pane.Markdown(f"""{e}""")
    try:
        df.DATE = pd.to_datetime(df.DATE,utc = True)
        # print(pd.to_datetime(df.DATE,utc = True))
        df["MONTHLY"] = df["DATE"].dt.strftime('%Y-%m')
        df['QUARTERLY'] = pd.PeriodIndex(df.DATE, freq='Q').astype(str)
        df['WEEKLY'] = pd.PeriodIndex(df.DATE, freq='W').astype(str)
    except Exception as e:
        return pn.pane.Markdown(f"""{e}""")    
    df = df.reset_index().rename(columns={df.index.name:'ID'}) #crate synthetic prediction ID for my code to run
    # df = df.dropna(subset = 'TARGET', axis = 1)
    df = df[~(df.TARGET.isna()) | (df.SCORE.isna())]
    if df.TARGET.nunique() > 2:
        df.TARGET = np.where(df.TARGET > 0 , 1 , 0)      
    df.SCORE = df.SCORE.astype(np.float64)

    # print(date_range_.value)

    # baselines
    try:
        baseline = df.set_index('MONTHLY').loc[date_range_.value[0]: date_range_.value[1]].reset_index().copy()
    except:
        baseline = df.copy()
        baseline = baseline.set_index('MONTHLY')
        baseline.index = pd.to_datetime(baseline.index)
        baseline = baseline.loc[date_range_.value[0]: date_range_.value[1]].reset_index()
        baseline["MONTHLY"] = baseline["MONTHLY"] .dt.strftime('%Y-%m')   
    #prods
    prod = df.loc[~df.MONTHLY.isin(list(baseline.MONTHLY.unique()))].copy()

    ##PSI##
    baseline_psi = baseline.copy()
    prod_psi = prod.copy()

    baseline_psi = add_extremes_OOT(baseline_psi, name = 'ID', score = 'SCORE')
    prod_psi["DEC_BANDS"] = pd.cut(prod_psi['SCORE'], bins = pd.qcut(baseline_psi['SCORE'],10, retbins = True)[1]) 
    prod_psi = prod_psi.groupby([period_metrics.value,
                                    "DEC_BANDS"]).agg(Count = ("DEC_BANDS",
                                    "count")).sort_index(level = 0).reset_index()
    prod_psi = prod_psi.groupby(period_metrics.value).apply(percentage).drop("Count",axis = 1)

    baseline_psi["DECILE"] = pd.cut(baseline_psi['SCORE'], bins = pd.qcut(baseline_psi['SCORE'],10, retbins = True)[1]) 
    baseline_psi = baseline_psi["DECILE"].value_counts()
    baseline_psi = baseline_psi / sum(baseline_psi)
    baseline_psi = baseline_psi.reset_index().rename(columns={"index":"DEC_BANDS", "DECILE": "percent"})
    baseline_psi[period_metrics.value] = "validation"
    baseline_psi = baseline_psi[[period_metrics.value, "DEC_BANDS", "percent"]]

    prod_psi = pd.concat([prod_psi,baseline_psi])

    prod_psi = prod_psi.pivot(index = "DEC_BANDS", columns=period_metrics.value)["percent"]
    psi_results = pn.widgets.DataFrame(psi(prod_psi).to_frame("%s_PSI"%period_metrics.value))

    baseline['QUARTERLY'] = 'Baseline: '+ baseline['QUARTERLY'].unique()[0] + '_' + baseline['QUARTERLY'].unique()[-1]
    baseline['MONTHLY'] = 'Baseline: '+ baseline['MONTHLY'].unique()[0] + '_' + baseline['MONTHLY'].unique()[-1]
    baseline['WEEKLY'] = 'Baseline: '+ baseline['WEEKLY'].unique()[0] + '_' + baseline['WEEKLY'].unique()[-1]

    auc_b = baseline.groupby([period_metrics.value]).apply(AUC)
    auc_p = prod.groupby([period_metrics.value]).apply(AUC)
    baseline_auc = pn.widgets.DataFrame(auc_b)
    prod_auc = pn.widgets.DataFrame(prod.groupby([period_metrics.value]).apply(AUC), width=600, height=1000, autosize_mode='fit_columns',name = 'AUC')
    return pn.Tabs(
        ('Metrics', pn.Column(
                    pn.Row(psi_results, save_csv(psi(prod_psi).to_frame("%s_PSI"%period_metrics.value), 'PSI')),
                    pn.Row(prod_auc, baseline_auc, save_csv(pd.concat([auc_b, auc_p], axis = 0), 'AUC')),
        sizing_mode='stretch_width'))
    )
        

    # return pn.Tabs(
    #         ('Analysis', pn.Column(
    #             pn.Row(vol_ret, pn.layout.Spacer(width=20), pn.Column(div, table), sizing_mode='stretch_width'),
    #             pn.Column(pn.Row(year, investment), return_curve, sizing_mode='stretch_width'),
    #             sizing_mode='stretch_width')),
    #         ('Timeseries', timeseries),
    #         ('Log Return', pn.Column(
    #             '## Daily normalized log returns',
    #             'Width of distribution indicates volatility and center of distribution the mean daily return.',
    #             log_ret_hists,
    #             sizing_mode='stretch_width'
    #         ))
    #     )

pn.Row(pn.Column(widgets), pn.layout.Spacer(width=30), run).servable()













# Caveats
# The maximum sizes set in either Bokeh or Tornado refer to the maximum size of the message that 
# is transferred through the web socket connection, which is going to be larger than the actual 
# size of the uploaded file since the file content is encoded in a base64 string. So if you set a
# maximum size of 100 MB for your application, you should indicate your users that the upload
# limit is a value that is less than 100 MB.

# When a file whose size is larger than the limits is selected by a user, their browser/tab may
# just crash. Alternatively the web socket connection can close (sometimes with an error message
# printed in the browser console such as [bokeh] Lost websocket 0 connection, 1009 (message too 
# big)) which means the application will become unresponsive and needs to be refreshed.

# app = ...

# MAX_SIZE_MB = 150

# pn.serve(
#     app,
#     # Increase the maximum websocket message size allowed by Bokeh
#     websocket_max_message_size=MAX_SIZE_MB*1024*1014,
#     # Increase the maximum buffer size allowed by Tornado
#     http_server_kwargs={'max_buffer_size': MAX_SIZE_MB*1024*1014}
# )

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