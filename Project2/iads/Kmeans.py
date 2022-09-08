import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#----------------------------------------------------------
def normalisation(df):
    return (df - df.min())/(df.max() - df.min())

#----------------------------------------------------------
def dist_euclidienne(v1 , v2):
    return np.sqrt(np.sum(np.square(v1-v2)))
#----------------------------------------------------------
def dist_manhattan(v1, v2):
    return (np.sum(np.abs(v1-v2)))
#----------------------------------------------------------
def dist_vect(v1, v2):
    return dist_euclidienne(v1,v2)
#------------------------------------
def centroide(df):
    return df.mean(axis=0)
#----------------------------------------------------------
def dist_centroides(grp1,grp2):
    return dist_vect(centroide(grp1),centroide(grp2))
#----------------------------------------------------------
def initialise(df):
    return dict(zip([i for i in range(len(df))],[[j]for j in range(len(df))]))
#----------------------------------------------------------
def inertie_globale(Base, U):
    inertie_globale = 0
    Base = np.array(Base)
    for i in U.keys():
        inertie_globale += inertie_cluster(Base[U[i]])
    return inertie_globale

def inertie_cluster(Ens):
    centro = centroide(Ens) #calcul du centroide
    inertie = 0
    for x in np.array(Ens):
        inertie += np.square(dist_vect(centro,x))
    return inertie

def init_kmeans(K,Ens):
    idx = np.random.choice(Ens.shape[0], K)
    Ens=np.array(Ens)
    return Ens[idx]

def plus_proche(Exe,Centres):
    return np.argmin([dist_vect(Exe,Centres[i]) for i in range(len(Centres))])

def affecte_cluster(Base,Centres):
    #U=dict(zip([i for i in range(len(Centres))],[[] for i in range(len(Centres))]))
    U = {}
    for i in range(len(Base)):
        index_centre = plus_proche(Base.iloc[i],Centres)
        if index_centre in U:
            U[index_centre].append(i)
        else:
            U[index_centre]= [i]
    return U
        
def nouveaux_centroides(Base,U):
    Base=np.array(Base)
    return np.array([list(centroide(Base[U[i]])) for i in U.keys()])


def affiche_resultat(Base,Centres,Affect):
    colors=['g','y','b']
    for i in range(len(Centres)) :
        plt.scatter(Base['X'][Affect[i]],Base['Y'][Affect[i]],color=colors[i])
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
        
        

def kmoyennes(K, Base, epsilon, iter_max):
    
    centres = init_kmeans(K,Base)#pour initiliser les centroides
    U  = affecte_cluster(Base,centres)
    centres = nouveaux_centroides(Base,U)
    old_iner_glob = inertie_globale(Base,U)
  
    for i in range(iter_max): 
        U = affecte_cluster(Base,centres)
        #mise à jour des centres
        centres = nouveaux_centroides(Base,U)
        new_iner_glob = inertie_globale(Base,U)
        diff = abs(new_iner_glob-old_iner_glob)
        if diff < epsilon:
            break
        old_iner_glob = new_iner_glob
    return (centres,U)

def dist_intra_cluster(Base,cluster):
    dist_max = 0
    Base=np.array(Base)
    for i in range(len(Base)):
        for j in range(len(Base)):
            dist=dist_vect(Base[i],Base[j])
            if (dist>dist_max):
                dist_max=dist
    return dist_max

def index_DUNN(Base,affect):
    #index_Dunn=Codist/inertie_globale
    codist=0
    for i in affect.keys():
        if len(affect[i])!=0:
            codist+=dist_intra_cluster(Base,affect[i])
    return codist/inertie_globale(Base,affect)
        
