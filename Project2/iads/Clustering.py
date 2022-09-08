import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


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

def fusionne(df,PO,verbose=False):
    dist_min = 1000
    dico = dict()
    for (k1,v1) in PO.items():
        for (k2,v2) in PO.items():
            if(k1!=k2):
                dist = dist_centroides(df.iloc[v1],df.iloc[v2])
                if (dist < dist_min):
                    dist_min = dist
                    k_fusion1 = k1
                    k_fusion2 = k2
        dico.update({(k_fusion1,k_fusion2):dist_min})
    #choisir le couple avec la plus petite distance
    (k_min1,k_min2)=min(dico, key=dico.get)
    #merge les deux clusters
    POC=PO.copy()
    POC.update({max(PO.keys())+1:PO[k_min1]+PO[k_min2]})
    del(POC[k_min1])
    del(POC[k_min2])
    if(verbose):
        print("Distance mininimale trouvée entre  [",k_min1,",", k_min2,"]  = ",dist_min)
    return (POC,k_min1, k_min2,dist_min)   




def clustering_hierarchique(df,verbose=False,dendrogramme=False):
    PO = initialise(df)
    PO_cop = PO.copy()
    res_merge = []
    while(len(PO)>1):
        (PO,k1,k2,dist) = fusionne(df,PO)
        if len(res_merge)==0:
            res_merge = [k1,k2,dist,len(PO_cop[k1])+len(PO_cop[k2])]
        else:
            res_merge= np.vstack([res_merge,[k1,k2,dist,len(PO_cop[k1])+len(PO_cop[k2])]])
        PO_cop = PO   
        if verbose:
            print("Distance mininimale trouvée entre  [",k1,",", k2,"]  = ",dist)
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(res_merge, leaf_font_size=24.)
        plt.show()
    
    return res_merge


def clustering_hierarchique_seuil(chaine, df, dist_seuil, verbose = True):
    courant = initialise(df)       # clustering courant, au départ:s données data_2D normalisées
    clusters = {i: {i} for i in range(df.shape[0])}
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes à fusionner
        #print(len(courant))
        novo, k1, k2, dist_min = fusionne(df,courant)
        if dist_min > dist_seuil:
            break
        clusters[max(max(clusters), k1, k2)+1] = clusters.pop(k1) | clusters.pop(k2)
        assert clusters.keys() == novo.keys()
        if(len(M_Fusion) == 0):
            M_Fusion = [k1, k2, dist_min, 2]
        else:
            M_Fusion = np.vstack([M_Fusion,[k1, k2, dist_min, 2]])
        courant = novo
    return clusters, courant

