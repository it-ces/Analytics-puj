# EDA...
import matplotlib.pyplot as plt
import seaborn as sns




def histogram_plot(df):
    fig, ax = plt.subplots(4,3, figsize=(12,13))
    dict = {i: df.select_dtypes(include=["float64"]).columns.to_list()[i] for i in range(12)} #Creación de diccionario para homologar llamado del ciclo iterativo
    k = 0
    for i in range(4):
        for j in range(3):

            sns.set_style("white")
            sns.histplot(ax = ax[i][j], data =  df, x=dict[k] , stat="density", common_norm=False, shrink = 0.9)
            k += 1


def violins(df):
        ### Histograma
    fig, ax = plt.subplots(4,3, figsize=(12,13))

    dict = {i: df.select_dtypes(include=["float64"]).columns.to_list()[i] for i in range(12)} #Creación de diccionario para homologar llamado del ciclo iterativo

    print(dict)
    k = 0

    for i in range(4):
        for j in range(3):

            sns.set_style("whitegrid")
            sns.violinplot(ax = ax[i][j], data =  df, x=dict[k])
            k += 1