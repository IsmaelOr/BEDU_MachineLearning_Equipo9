import pandas as pd

from scipy import stats
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis

import pprint as pp

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

import plotly.express as px

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
import sklearn.cluster as cluster

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn import tree as treeClassifier

from sklearn.ensemble import RandomForestClassifier as forest

from sklearn.linear_model import SGDClassifier

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix

#import chart_studio.plotly as py
import plotly.express as px

from sklearn.tree import DecisionTreeRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

#from dataprep.eda import create_report

from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler,PolynomialFeatures,PowerTransformer,OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.stats as st

from jcopml.plot import plot_correlation_matrix,plot_classification_report,plot_confusion_matrix,plot_residual,plot_association_matrix
from jcopml.tuning import grid_search_params as gsp, random_search_params as rsp, bayes_search_params as bsp
#from jcopml.feature_importance import mean_score_decrease
from skopt import BayesSearchCV
from jcopml.tuning.space import Integer,Real

from sklearn.metrics import plot_confusion_matrix

from dataprep.eda import create_report

import cv2
from skimage import io

# La función 'particionar' se encarga de separar el Dataset en grupos de entrenamiento y prueba.
def particionar(entradas, salidas, porcentaje_entrenamiento, porcentaje_prueba):
    x_train, x_test, y_train, y_test = train_test_split(entradas, salidas, test_size = porcentaje_prueba)
    return [x_train, x_test, y_train, y_test]

# La función 'kfold_function' se encarga de hacer divisiones para entregar varias muestras.
def kfold_function(data_x, k):
  muestras = []
  n = []
  random_seed = 48
  kfold = KFold(n_splits = k, shuffle = True, random_state= random_seed)
  
  for indices_train, indices_test in kfold.split(data_x):
    n.append(indices_train)
    n.append(indices_test)
    muestras.append(n)
    n = []

  return muestras

# La función 'LOOCV_function' separa al Dataset en la misma cantidad de registros del mismo.
def LOOCV_function(data_x):
  muestras = []
  n = []
  random_seed = 48
  kfold = KFold(n_splits= data_x.shape[0], shuffle= True, random_state= random_seed)
  
  for indices_train, indices_test in kfold.split(data_x):
    n.append(indices_train)
    n.append(indices_test)
    muestras.append(n)
    n = []

  return muestras

# La función 'Matrix_confusion' entrega los resultados de 'exactitud', 'sensibilidad', 'especificidad' y 'precisión' visualizados a través de una matriz de confusión.
def Matrix_confusion(y_esperados, y_predichos):
  resultado = confusion_matrix(y_esperados, y_predichos)
  print(resultado)

  (TP, FN, FP, TN) = resultado.ravel()

  print("\nTrue positives: "+str(TP))
  print("True negatives: "+str(TN))
  print("False positives: "+str(FP))
  print("False negative: "+str(FN))

  accuracy = (TP + TN) * 100 / (TP + TN + FP + FN)
  sensibilidad = TP * 100 / (TP + FN)
  especificidad = TN * 100 / (TN + FP)
  precision = (TP) * 100 / (TP + FP)

  print("\nExactitud: " + str(accuracy) + "%")
  print("Sensibilidad: " + str(sensibilidad) + "%")
  print("Especificidad: " + str(especificidad) + "%")
  print('Precisión: ' + str(precision) + '%')

# La función 'centroideCercano' ayuda a la muestra en cuestión a encontrar el centroide que le pertenece.
def centroideCercano(muestra, listaCentroides):
    listaDistancias = distEuclidiana(muestra, listaCentroides)
    centroideCercano = np.argmin(listaDistancias)
    return centroideCercano

# La función 'distEuclidiana' calcula la distancia entre dos datos.
def distEuclidiana(muestra, dataset):
    distancias = np.zeros((dataset.shape[0], 1))
    for counter in range(0,dataset.shape[0]):
        distancias[counter] = np.linalg.norm(muestra-dataset[counter])
    return distancias

# La función 'clasificarPorCentroides' divide el Dataset en diferentes centroides.
def clasificarPorCentroides(muestras, centroides):
    resultado = np.zeros((muestras.shape[0], 1))
    for counter in range(0, muestras.shape[0]):
        resultado[counter] = centroideCercano(muestras[counter], centroides)
    return resultado

def convertirAGrayScale(imagen):
    imagenGris = np.sum(imagen, axis = 2) / 3    
    return imagenGris

def binarizar(imagenGris, threshold):
    imgBinaria = np.where(imagenGris > threshold, 255, 0)
    return imgBinaria

def reducirColores(imagenGris, cantidadDeColores):
    if(cantidadDeColores <= 0):
        return np.zeros_like(imagenGris)

def obtenerNegativo(imagen):
    negativo = np.abs(imagen - 255)
    return negativo

def crearHistograma(imagen):
    histograma = np.zeros((256))
    imgEnArray = np.ravel(imagen)
    for counter in range(0, len(imgEnArray)):
        histograma[int(imgEnArray[counter])]+=1
        
    return histograma
