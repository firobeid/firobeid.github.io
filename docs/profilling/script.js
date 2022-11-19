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
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.0/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.0/dist/wheels/panel-0.14.0-py3-none-any.whl', 'holoviews>=1.15.1', 'holoviews>=1.15.1', 'hvplot', 'numpy', 'pandas']
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
import io
import sys
import os
import pandas as pd

import gc #garabage collector
from io import BytesIO
import panel as pn
import holoviews as hv
import hvplot.pandas
from warnings import filterwarnings


filterwarnings("ignore")
hv.extension('bokeh')


text = """
#  Feature Distribution and Stats
## AUTHOR: \`FIRAS ALI OBEID\`

This tool performs feature binning by equal intervals and by equal pouplations in each interval vs bad rate
To get the feature deep dive feature distribution:

1. Upload a CSV (only numerical data)

2. Choose & press on the binary (0 / 1) target column in the \`Select Target Variable\` section below

3. Press Run Analysis

4. Wait few seconds and analyze the updated charts
"""

file_input = pn.widgets.FileInput(align='center')
selector = pn.widgets.MultiSelect(name='Select Target Variable')
button = pn.widgets.Button(name='Run Analysis')
widgets = pn.WidgetBox(
    pn.panel(text, margin=(0, 10)),
    pn.panel('Upload a CSV containing  (X) features and  (y) binary variable:', margin=(0, 10)),
    file_input,
    selector,
    button
)


def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def get_data():
    global target, New_Refit_routing
    if file_input.value is None:
        New_Refit_routing = pd.DataFrame({"Open_accounts": np.random.randint(1,50,100000),
                                          "Income": np.random.randint(1000,20000,100000),
                                          "Years_of_experience": np.random.randint(0,20,100000),
                                          "default": np.random.random_integers(0,1,100000)})
        target = "default"
    else:
        New_Refit_routing = BytesIO()
        New_Refit_routing.write(file_input.value)
        New_Refit_routing.seek(0)
        try:
            New_Refit_routing = pd.read_csv(New_Refit_routing, error_bad_lines=False)#.set_index("id")
        except:
            New_Refit_routing = pd.read_csv(New_Refit_routing, error_bad_lines=False)
        target = None
        New_Refit_routing = New_Refit_routing.select_dtypes(exclude=['datetime', "category","object"])
        New_Refit_routing = New_Refit_routing[[cols for cols in New_Refit_routing.columns if New_Refit_routing[cols].nunique() >= 2]] #remove columns with less then 2 unique values
    return target, New_Refit_routing


def update_target(event):
    _ , New_Refit_routing = get_data()
    target = list(New_Refit_routing.columns)
    selector.set_param(options=target, value=target)

file_input.param.watch(update_target, 'value')
update_target(None)



def stats_():
    global stats
    stats = New_Refit_routing.describe().T
    stats["Missing_Values(%)"] = (New_Refit_routing.isna().sum() / len(New_Refit_routing)) * 100
    stats = stats.round(4).astype(str)

def cuts_(target):
    global test, test2, final_df , outlier_removed_stats
    df = New_Refit_routing.copy() 
    neglect = [target] + [cols for cols in df.columns if df[cols].nunique() <= 2] #remove binary and target variable
    cols = df.columns.difference(neglect)  # Getting all columns except the ones in []
    #REMOVE OUTIERS#
    df[cols] = df[cols].apply(lambda col: col.clip(lower = col.quantile(.01), 
                                        upper = closest(col[col < col.quantile(.99)].dropna().values, 
                                        col.quantile(.99))),axis = 0)
    outlier_removed_stats = df.describe().T
    remove_feature = list(outlier_removed_stats[(outlier_removed_stats["mean"]==outlier_removed_stats["max"]) & 
                        (outlier_removed_stats["mean"]==outlier_removed_stats["min"])].index)
    outlier_removed_stats = outlier_removed_stats.round(4).astype(str)

    neglect += remove_feature
    cols = df.columns.difference(neglect)  # Getting all columns except the ones in []

    
    df[cols] = df[cols].apply(lambda col: pd.cut(col.fillna(np.nan),
                                                bins = pd.interval_range(start=col.min(), end=col.max(), 
                                                periods = 10), include_lowest=True).cat.add_categories(pd.Categorical(f"Missing_{col.name}")).fillna(f"Missing_{col.name}"), axis=0)


    test = pd.concat([df[cols].value_counts(normalize = True) for cols in df[cols]], axis = 1)
    cols = test.columns
    test = test.reset_index().melt(id_vars="index", 
                                var_name='column', 
                                value_name='value').dropna().reset_index(drop = True)


    test = test.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Count_Pct"})
    test.Count_Pct = test.Count_Pct.round(4)
    test.IntervalCuts = test.IntervalCuts.astype(str)
    test.IntervalCuts = test.IntervalCuts.apply(lambda x: "("+str(round(float(x.split(",")[0].strip("(")),4)) +', ' + str(round(float(x.split(",")[-1].strip("]")),4)) +"]" if (x.split(",")[0].strip("(")[0]).isdigit() else x)

    test2 = pd.concat([df.groupby(col)[target].mean().fillna(0) for col in df[cols]], axis = 1)
    test2.columns = cols
    test2 = test2.reset_index().melt(id_vars="index", var_name='column', value_name='value').dropna().reset_index(drop = True)
    test2 = test2.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Bad_Rate_Pct"})
    test2.Bad_Rate_Pct = test2.Bad_Rate_Pct.round(4)
    test2.IntervalCuts = test2.IntervalCuts.astype(str)
    test2.IntervalCuts = test2.IntervalCuts.apply(lambda x: "("+str(round(float(x.split(",")[0].strip("(")),4)) +', ' + str(round(float(x.split(",")[-1].strip("]")),4)) +"]" if (x.split(",")[0].strip("(")[0]).isdigit() else x)


    test["index"] = test["feature"] + "_" + test["IntervalCuts"]
    test = test.set_index("index")
    test2["index"] = test2["feature"] + "_" + test2["IntervalCuts"]
    test2 = test2.set_index("index")
    final_df = pd.merge(test2, test[test.columns.difference(test2.columns)], on = "index")
   

## QCUT ##
def qcuts_(target):
    global test_q, test2_q, final_df_q
    df2 = New_Refit_routing.copy()
    neglect = [target] + [cols for cols in df2.columns if df2[cols].nunique() <= 2] #remove binary and target variable
    cols = df2.columns.difference(neglect)  # Getting all columns except the ones in []

    #REMOVE OUTIERS#
    df2[cols] = df2[cols].apply(lambda col: col.clip(lower = col.quantile(.01), 
                                        upper = closest(col[col < col.quantile(.99)].dropna().values, 
                                        col.quantile(.99))),axis = 0)
    temp = df2.describe().T
    remove_feature = list(temp[(temp["mean"]==temp["max"]) & 
                        (temp["mean"]==temp["min"])].index)

    neglect+= remove_feature
    cols = df2.columns.difference(neglect)  # Getting all columns except the ones in []
    # rank(method='first') is a must in qcut 
    # df2[cols] = df2[cols].apply(lambda col: pd.qcut(col.fillna(np.nan).rank(method='first'),
    #                                                 q = 10, duplicates = "drop").cat.add_categories(pd.Categorical(f"Qcut_Missing_{col.name}")).fillna(f"Qcut_Missing_{col.name}"), axis=0)
    df2[cols] = df2[cols].apply(lambda col: pd.qcut(col.fillna(np.nan).rank(method='first'),q = 10, labels=range(1,11)).cat.rename_categories({10:"Last"}).astype(str).replace(dict(dict(pd.concat([col,
           pd.qcut(col.fillna(np.nan).rank(method='first'),q = 10, labels=range(1,11)).cat.rename_categories({10:"Last"})
           .apply(str)], axis = 1, keys= ["feature", "qcuts"]).groupby("qcuts").agg([min, max]).reset_index().astype(str).set_index("qcuts",drop = False)
     .apply(lambda x :x[0]+"_"+"("+x[1]+","+x[2]+"]",axis = 1)),**{"nan":f"Qcut_Missing_{col.name}"})), axis=0)

    test_q = pd.concat([df2[cols].value_counts(normalize = True) for cols in df2[cols]], axis = 1)
    cols = test_q.columns
    test_q = test_q.reset_index().melt(id_vars="index", 
                                var_name='column', 
                                value_name='value').dropna().reset_index(drop = True)


    test_q = test_q.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Count_Pct"})
    test_q.Count_Pct = test_q.Count_Pct.round(4)
    test_q.IntervalCuts = test_q.IntervalCuts.astype(str)


    test2_q = pd.concat([df2.groupby(col)[target].mean().fillna(0) for col in df2[cols]], axis = 1)
    test2_q.columns = cols
    test2_q = test2_q.reset_index().melt(id_vars="index", var_name='column', value_name='value').dropna().reset_index(drop = True)
    test2_q = test2_q.rename(columns={"index":"IntervalCuts", "column":"feature", "value":"Bad_Rate_Pct"})
    test2_q.Bad_Rate_Pct = test2_q.Bad_Rate_Pct.round(4)
    test2_q.IntervalCuts = test2_q.IntervalCuts.astype(str)

    test_q["index"] = test_q["feature"] + "_" + test_q["IntervalCuts"]
    test_q = test_q.set_index("index")
    test2_q["index"] = test2_q["feature"] + "_" + test2_q["IntervalCuts"]
    test2_q = test2_q.set_index("index")
    final_df_q = pd.merge(test2_q, test_q[test_q.columns.difference(test2_q.columns)], on = "index")
    



@pn.depends(button.param.clicks)
def run(_):
    target, New_Refit_routing = get_data()
    if target == None:
        target = str(selector.value[0])
    else:
        target = "default"
    print(str(selector.value[0]))
    print(target)
    # print(type(file_input.value))
    # print(type(New_Refit_routing))
    print(New_Refit_routing.head())

    stats_()
    cuts_(target)
    qcuts_(target)
    test2_plot = test2.set_index("IntervalCuts").hvplot.scatter(yaxis = "left", y = "Bad_Rate_Pct",
            groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
            width = 1000, title = "Features Segments Cuts by Count", legend = True,label = "Bad Rate(%)").opts(xrotation=45, yformatter = "%.04f",show_grid=True, 
                                                                                        framewise=True, color = "red", legend_position='top_right')
    test_plot = test.set_index("IntervalCuts").hvplot.bar(y = "Count_Pct",
                groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
                width = 1000, title = "Features Segments Cuts by Count", legend=True, alpha=0.3, label ="Equal Intervals Data Points(%)").opts(xrotation=45, yformatter = "%.04f",show_grid=True, framewise=True, yaxis='left')
    final_table = final_df.hvplot.table(groupby = "feature", width=400)

    test2_plot_q = test2_q.set_index("IntervalCuts").hvplot.scatter(yaxis = "left", y = "Bad_Rate_Pct",
                groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
                width = 1000, title = "Features Segments Q_Cuts by Count", legend = True).opts(xrotation=45, yformatter = "%.04f",show_grid=True, 
                                                                                                framewise=True, color = "red")
    test_plot_q = test_q.set_index("IntervalCuts").hvplot.bar(y = "Count_Pct",
                groupby = "feature", xlabel = "Intervals(Bins)", ylabel = "%Count vs %BadRate",height = 500,
                width = 1000, title = "Features Segments Q_Cuts by Count", legend=True, alpha=0.3, label ="Equal Population Data Points(%)").opts(xrotation=45, yformatter = "%.04f",show_grid=True, framewise=True, yaxis='left')
    final_table_q = final_df_q.hvplot.table(groupby = "feature", width=400)


    stats_table = stats.reset_index().hvplot.table(width = 1000,title="Summary Statistics of the Data", hover = True, responsive=True, 
                    shared_axes= False, fit_columns = True,
                    padding=True, height=500, index_position = 0, fontscale = 1.5)
    stats_table_no_outliers = outlier_removed_stats.reset_index().hvplot.table(width = 1000,title="Summary Statistics of the Capped Outliers Data", hover = True, responsive=True, 
                    shared_axes= False, fit_columns = True,
                    padding=True, height=500, index_position = 0, fontscale = 1.5)
    #PANEL
    pn.extension( template="fast")
    pn.state.template.param.update(
        # site_url="",
        # site="",
        title="Feature Distribution & Statistics",
        # favicon="",
    )
    
    title = pn.pane.Markdown(
    """
    ### Feature Distribution (Bin Count & Bad Rate)
    """,
    width=800,
    )

    return pn.Column(
                title,
                (test2_plot * test_plot * test2_plot_q * test_plot_q + (final_table + final_table_q)).cols(3),
                (stats_table + stats_table_no_outliers).cols(2),
            )




pn.Row(pn.Column(widgets), pn.layout.Spacer(width=20), run).servable(target='main')

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