# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2022

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # Extraction des exemples de classe -1:
    data_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data_positifs = desc[labels == +1]
    # Affichage de l'ensemble des exemples :
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1
    
# ------------------------ 
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
	
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data_desc = np.random.uniform(inf,sup,(n*p,p))#X
    data_label = np.asarray([-1 for i in range(0,n)] + [+1 for i in range(0,n)])#Y
    return (data_desc,data_label)
	
# ------------------------ 
def genere_dataset_gaussian(positive_center,positive_sigma,negative_center,negative_sigma,nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    X_negatifs_red = np.random.randn(nb_points,2)#classe -1 loi nomale centrée reduite
    X_negatifs = (X_negatifs_red * negative_sigma) + negative_center
    X_positifs_red = np.random.randn(nb_points,2)#classe 1
    X_positifs = (X_positifs_red * positive_sigma) + positive_center
    data_label_gauss = np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
    data_gauss_desc = np.vstack((X_negatifs,X_positifs))
    return (data_gauss_desc , data_label_gauss)
# ------------------------ 
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    data_gauss_desc1,data_gauss_label1 =     genere_dataset_gaussian(np.array([1,1]),math.sqrt(sigma_positif),np.array([0,0]),math.sqrt(sigma_positif),n)
    data_gauss_desc2, data_gauss_label2 = genere_dataset_gaussian(np.array([0,1]),math.sqrt(sigma_positif),np.array([1,0]),math.sqrt(sigma_positif),n)
    return (np.concatenate((data_gauss_desc1,data_gauss_desc2)),np.concatenate((data_gauss_label1[0:n],data_gauss_label2[0:n],data_gauss_label1[n:],data_gauss_label2[n:])))

def crossval(X, Y, n_iterations, iteration):
    """extrait 02 sous data sets un d'apprentissage et l'autre de test"""
    i_deb_test = iteration*(len(X)//n_iterations)
    i_fin_test = (iteration+1)*(len(X)//n_iterations)-1
    Ytest = Y[i_deb_test:i_fin_test+1]
    Xtest = X[i_deb_test:i_fin_test+1]
    Xapp = np.concatenate((X[0:i_deb_test],X[i_fin_test+1:len(X)]))
    Yapp = np.concatenate((Y[0:i_deb_test],Y[i_fin_test+1:len(X)]))
    return Xapp, Yapp, Xtest, Ytest
#-------------------------------------------------
def crossval_strat(X, Y, n_iterations, iteration):
    #classe 1
    w1 = np.where(Y==1)[0]
    Y1 = Y[w1]
    X1 = X[w1]
    #classe -1
    w_1 = np.where(Y==-1)[0]
    Y_1 = Y[w_1]
    X_1 = X[w_1]
    Xapp1, Yapp1, Xtest1, Ytest1 = crossval(X1,Y1,n_iterations,iteration)
    Xapp2, Yapp2, Xtest2, Ytest2 = crossval(X_1,Y_1,n_iterations,iteration)
    return np.concatenate((Xapp2,Xapp1)), np.concatenate((Yapp2,Yapp1)),       np.concatenate((Xtest2,Xtest1)),np.concatenate((Ytest2,Ytest1))

