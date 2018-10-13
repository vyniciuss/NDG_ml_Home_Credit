# ================================================== #
#               Helper Functions                     #
# ================================================== #
import sys
import pandas as pd
import plotly.offline as po
po.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.plotly as py
from plotly import tools
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

def draw_bar_plot_vert(col_name, data):    
    tempAll = data[col_name].value_counts()
    temp1 = data[data["TARGET"] == 1][col_name].value_counts()
    temp0 = data[data["TARGET"] == 0][col_name].value_counts()
    
    trace0 = go.Bar(x=tempAll.index, y=(tempAll / tempAll.sum())*100, name='All', marker=dict(color="#49b675"))
    trace1 = go.Bar(x=temp1.index, y=(temp1 / temp1.sum())*100, name='Target = 1', marker=dict(color="#e61919"))
    trace2 = go.Bar(x=temp0.index, y=(temp0 / temp0.sum())*100, name='Target = 0', marker=dict(color="#87cefa"))
    return trace0, trace1, trace2 

def missing_values(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percentual = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).round(1)
    missing_data  = pd.concat([total, percentual], axis=1, keys=['Total de Ausentes', 'Percentual de Ausentes'])
    return missing_data

def print_percentuals_default(df, column, ruido):
    outlier = df[df[column] == ruido]
    no_outlier = df[df[column] != ruido]
    print('Registros diferentes do valor informado cometeram calote em %0.2f%% dos empréstimos' % (100 * no_outlier['TARGET'].mean()))
    print('Registros iguais ao valor informado cometeram calote em  %0.2f%% dos empréstimos' % (100 * outlier['TARGET'].mean()))
    print('Existem {} valores iguais ao informado "{}" no conjunto de dados'.format(len(outlier), ruido))