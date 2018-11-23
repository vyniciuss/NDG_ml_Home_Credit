# ================================================== #
#               Helper Functions                     #
# ================================================== #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn import cross_validation
init_notebook_mode(connected=True)
import warnings



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

def draw_bar_plot_hor(x, y, titulo):
    data = [go.Bar(x=x, y=y, marker=dict(color=x, colorscale = 'Jet', reversescale = True), name=titulo, orientation='h')]
    layout = go.Layout(title=titulo, width = 900, height = 4000, margin=dict(l=300), yaxis=dict(showgrid=True))
    fig1 = go.Figure(data=data, layout =layout)
    iplot(fig1)

def draw_features_most_correlated(df, feature):

    cor =  df.corr()[feature].sort_values()
    features = []
    scores = []
    for index, val in cor.tail(6).iteritems():
        if index == feature or math.isnan(val):
            continue
        features.append(index)
        scores.append(val)
            
    for index, val in cor.head(5).iteritems():
        features.append(index)
        scores.append(val) 

    correlacao = pd.DataFrame({'features' : features, 'coeficiente' : scores}).set_index('features').T
    correlacao.plot(kind="bar", figsize = (10, 8))
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2.,
                                            p.get_height()), ha='center',
                    va='center', xytext=(0, 10), textcoords='offset points')
    ax.tick_params(axis = 'x', which = 'major', pad = 15, size = 42)
    ax.set_title('Correlação com a feature {}'.format(feature))
    ax.set_ylabel('Coeficiente')
    ax.set_xlabel('Features')
    plt.setp(ax.get_xticklabels(), rotation = 0)
    plt.show()

def draw_grupos_idade_por_situacao(group, colum, labels_grupos):
 
    year_target_group = group.groupby(['YEARS_BINNED','TARGET']).mean()
    #transoformando os index YEARS_BINNED e TARGET em colunas 
    year_target_group.reset_index(level=1, inplace=True)
    group1 = year_target_group[year_target_group['TARGET'] == 1][colum]
    group0 = year_target_group[year_target_group['TARGET'] == 0][colum]

    trace0 = go.Scatter(
        x = labels_grupos,
        y = group1,
        mode = 'lines',
        name = 'Inadimplentes',
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 3
        )
    )

    trace1 = go.Scatter(
        x = labels_grupos,
        y = group0,
        mode = 'lines',
        name = 'Bons Pagadores',
        line = dict(
            color = ('rgb(22, 96, 167)'),
            width = 2)
    )
    return trace0, trace1

def draw_roc_score(classifier, x_train, y_train, x_test, y_test):
    #train
    y_probas_train = classifier.predict_proba(x_train)
    y_scores_train = y_probas_train[:, 1] 
    predictions_train = classifier.predict(x_train)
    cv_score_train = cross_validation.cross_val_score(classifier, x_train, y_train, cv=5, scoring='roc_auc')
    print("Train Accuracy: {}%".format(accuracy_score(y_train, predictions_train)))
    print("Train Score Roc: {}%".format(np.round(roc_auc_score(y_train, y_scores_train)*100, 3)))
    print("Train CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score_train),
                                                                                   np.std(cv_score_train),
                                                                                   np.min(cv_score_train),
                                                                                   np.max(cv_score_train)))
    print("------------------------------------------------------------------------------------------------")
    #teste
    y_probas_test = classifier.predict_proba(x_test)
    y_scores_test = y_probas_test[:, 1] 
    predictions_test = classifier.predict(x_test)
    cv_score_test = cross_validation.cross_val_score(classifier, x_test, y_test, cv=5, scoring='roc_auc')
    print("Test Accuracy: {}%".format(accuracy_score(y_test, predictions_test)))
    print("Test Score Roc: {}%".format(np.round(roc_auc_score(y_test, y_scores_test)*100, 3)))
    print("Test CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score_test),
                                                                                   np.std(cv_score_test),
                                                                                   np.min(cv_score_test),
                                                                                   np.max(cv_score_test)))
    #draw roc curve
    fpr1, tpr1, thresholds = roc_curve(y_train, y_scores_train)
    fpr2, tpr2, thresholds = roc_curve(y_test, y_scores_test)
    plt.title("Score Roc em Treino e Teste")    
    line1, = plt.plot(fpr1, tpr1, 'b', label="Treino", linewidth=1.5)
    line2, = plt.plot(fpr2, tpr2,'r', label="Teste", linewidth=1.5)
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.show()

def draw_feature_importance(feature_importances, columns):
    
    x, y = (list(x) for x in zip(*sorted(zip(feature_importances, columns), 
                                                            reverse = False)))
    draw_bar_plot_hor(x, y, 'Importância das features')