# ================================================== #
#               Helper Functions                     #
# ================================================== #
import pandas as pd
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
init_notebook_mode(connected=True)
import warnings
from sklearn.model_selection import StratifiedShuffleSplit

def draw_bar_plot_vert(col_name, df):    
    tempAll = df[col_name].value_counts()
    temp1 = df[df["TARGET"] == 1][col_name].value_counts()
    temp0 = df[df["TARGET"] == 0][col_name].value_counts()
    
    trace0 = go.Bar(x=tempAll.index, y=(tempAll / tempAll.sum())*100, name='All', marker=dict(color="#49b675"))
    trace1 = go.Bar(x=temp1.index, y=(temp1 / temp1.sum())*100, name='Target = 1', marker=dict(color="#e61919"))
    trace2 = go.Bar(x=temp0.index, y=(temp0 / temp0.sum())*100, name='Target = 0', marker=dict(color="#87cefa"))
    return trace0, trace1, trace2 

def missing_values(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percentual = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False).round(1)
    missing_data  = pd.concat([total, percentual], axis=1, keys=['Total de Ausentes', 'Percentual de Ausentes'])
    return missing_data

def print_percentuals_default(df, column, ruido):
    outlier = df[df[column] == ruido]
    no_outlier = df[df[column] != ruido]
    print('Registros diferentes do valor informado cometeram calote em %0.2f%% dos empréstimos' % (100 * no_outlier['TARGET'].mean()))
    print('Registros iguais ao valor informado cometeram calote em  %0.2f%% dos empréstimos' % (100 * outlier['TARGET'].mean()))
    print('Existem {} valores iguais ao informado "{}" no conjunto de dados'.format(len(outlier), ruido))
    
def draw_count_uniques(col_name, df, titulo):
    temp = df[col_name].astype('category').map({0:'NÃO',1:'SIM'})
    count_uniques = temp.value_counts()
    data = [go.Bar(x=count_uniques.index, y=(count_uniques / count_uniques.sum())*100, name='All', marker=dict(color=['rgb(49,130,189)', 'rgb(204,204,204)']))]
    layout = go.Layout(title=titulo)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    print("Total de itens em cada grupo:")
    print(temp.value_counts())

def nomes_atributos_categoricos(df):
    return [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]

def nomes_atributos_numericos(df):
    return [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64', 'int64']]
