#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:15:57 2022

@author: pierretramoni
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats



def markov_chain(n,P,nu,nu_iter=int(2)):
    """
    

    Parameters
    ----------
    n : int
        lenght of the Markov chain simulation
    P : Array(k,k)
        Transition Matrix
    nu : vector(1,k)
        Initial weightned probabilistic matrices
    nu_iter : int, optional
        Number of the iteration of the markov chain The default is int(2).

    Returns
    -------
    X : Array(n,nu_iter)
        it's the path of the markov chain from the begining
        the init!!!!!!!!!!!!!

    """
    "simule les n premiers pas d'une chaine de Markov "
    "à partir de sa matrice de transition"
    "creation d'un random generator depending on the probability to be in a state i at time 0 "
    " the output will be an integer from 0 to the number of state in our matrix weighted  by the depending probability(nu)"
    xk = np.arange(len(P)).reshape((-1,1))
    custm = stats.rv_discrete(name='custm', values=(xk, nu.T))
    R_nu = custm.rvs(size=nu_iter)
    X=np.zeros((n,len(R_nu)),dtype=np.int) #X can only takes integer values.
    X[0]= R_nu   
    for k in range(n-1):
        for i in range(len(R_nu)):
            X[k+1][i]=np.random.choice(a=range(len(P)), p=P[X[k][i],:])
        # Les états sont numérotés de 0 à len(P)-1   
    return X



    

### statistique
## pour calc l'absorbtion il faut une condition sur la diga==1 de la mat si 
# elle existe alors il ya aura probablement absorbtion apres a refdlchir comment faire

def statistique_markov_chain(n,P,nu_iter,X):
    """

    Parameters
    ----------
    X : Array(n,nu_iter)
        return of the function markov_p which is the simulation of 
        the step by step transition matrix 
    P : Array(i,i)
        transition matrix
        DESCRIPTION.
    n : integer ()
        number of step we want

    Returns
    -------
    times_in_state : (nu_iter,n) Array
        return the number of times we are at states i through the length of the chain
    times_means : (nu_iter,n)
        return the probability to be in a states i through the length of the chain
    y : (n,nu_iter,i) 3d Array
        tensor of each probability to be in a state i over times for nu_iteration
    lst : (nu_iter) list
        return the first rank of the chain where we are in an absorbing state

    """
    y=np.zeros((n,nu_iter,len(P)))
    times_in_state = np.zeros((nu_iter,len(P)))
    times_means = np.zeros((nu_iter,len(P)))
    for j in range(nu_iter):        
        for i in range(len(P)):
           
            (X[:,j]==i) #donne un vecteur de booleen
            times_in_state[j,i] = ((X[:,j]==i).sum()) #donne le nombre de fois ou l'on est passe dans l'etat 0
            times_means[j,i]=((X[:,j]==i).mean()) #donne la moyenne empirique
            y[:,j,i] = ((X[:,j]==i).cumsum()/np.arange(1,len(X)+1))

    
    # compute for an absorbing matrices the first time than it's in an absorbing states
    # we wanna see if the matrix is an absorbing matrix :
    any_ones = 1 in [P[i,i] for i in range(len(P))]
    lst=[]
    if (any_ones  == True):
        loc = np.where(np.diag(P)==1)
        print("\n it's an absorbing matrices, and the absorbing state is :", loc[0])

        #we should looks for the first value of the rank absorbing in the X 
        # collect all the rank of the absorb mat
        fv = [np.where(X[:,i]==loc[0]) for i in range(nu_iter)]
        # see how to improve the code for the case there is more than 1 absorbing state
        for i in range(len(fv)):
            try: # I did this cause some times I didn't raech an absorbing states 
                lst.append(fv[i][0][0])
            except:
                pass
     
    return times_in_state, times_means, y, lst

def Average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    markov_chain()
    statistique_markov_chain()
    Average()



