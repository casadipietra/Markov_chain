import numpy as np
from numpy.linalg import inv,  lstsq, matrix_power
from fractions import Fraction
#import matplotlib.pyplot as plt 
# Import the MarkovChain class from markovchain.py



def Matrix(*args, **kwargs):
    """
    Matrix()
    input the matrix which is supposed to be a transition matrix

    Returns
    -------
    n : int
        dimension of the matrix P : (n*n) .
    mat : (i,j) Array
        DESCRIPTION.
        
    # Thinkabout change n in too an integer cause the matrix P is always a square matrix 
    print("\n Enter the dimension of the matrix [n,n]")
    
    n = int(input("n : "))
        mat = [[0 for j in range(n)] for i in range(n)]
    
    """
    print("\n Enter the dimension of the matrix [i,j]")
    n = int(input("n : "))
    mat = [[0 for j in range(n)] for i in range(n)]

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j]=float(Fraction(input("Input element ["+ str(i+1)+"," + str(j+1)+ "] : ")))
            mat = np.array(mat)
            print("\n Your Matrix is : \n",mat)
            
    
    return n, mat


def Stochastic_matrix(mat, n):
    """
    Stochastic_matrix(n, mat) we wanna check if the matrix is a stochastic matrix
    so it's verifie that each element of the matrix is superior to 0
    and if the sum of each row is equal to 1 
    
    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    mat : TYPE
        DESCRIPTION.

    Returns
    -------
    mat_sto : (i,1)
        should be equal to 1 for each row.

    """
    
    if (((mat>=0) & (mat<=1))==True).all :
        mat_sto = sum([mat[:,k] for k in range(n)]).reshape((-1,1,))
    
        if np.array_equal(np.round(sum([mat[:,k] for k in range(n)]).reshape((-1,1,)),12),np.ones((n,1))):
        #if ([sum([mat[:,k] for k in range(n[1])]).reshape((-1,1,))] != np.ones((n[1],1))).any:
            print("\n it's a Markov chain")
            #mc = MarkovChain(mat, list(range(1,len(mat)+1)))
            return mat_sto
        else:
            print("\n it's not a Markov chain the sum of each row are between 0 and 1")
    else:
        print("\n it's not a Markov chain cause (one or more) élément(s) are negative or strictly above 1")


def type_mat(mat,n):
    #look for improve any_ones loop
    any_ones = 1 in [mat[i,i] for i in range(n)]
    if (any_ones  == True):
        print("it's an absorbing Markov chain")
        absorbing = True       
        regular = False
        irreducible = False
    else:
        print("not an absorbing chain")
        absorbing = False
    
        for i in range(n*2):
            if ((matrix_power(mat,i)>0).all() == False):
                i += 1   
                print("the matrix is not regular at the rank :", i )
                regular = False
                if (matrix_power(mat-np.identity(n),n-1)>0).all() == True:
                    irreducible = True
                else:
                    irreducible = False

            else:
                print( "the matrix is regular at rank :", i, "so it's also an irreductible matrix ")
                "the Matrix P is regular (it is ergodic) => irreducible"
                regular = True
                irreducible = True
                break
    if (regular == False):
        print("the matrix is not regular at any rank :")

        
    
    return absorbing, regular, irreducible



def PVNu(mat,n,nu,k, absorbing = True):
    """
    

    Parameters
    ----------
    mat : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    nu : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    absorbing : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    PVn : TYPE
        DESCRIPTION.

    """
    
    PVn = nu @ matrix_power(mat,k)
    print("P_nu(X_",k,"=i) = ", PVn, sep='')
    return PVn
    
def transition_split(n,mat,nu, absorbing = True):
    """
    only for absorbing matrix
    
        [Q   R]
    P = [0   I]
        

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    mat : TYPE
        DESCRIPTION.

    Returns
    -------
    I : (r,r) Array
        DESCRIPTION.
    Q : (n-r,n-r) Array
        DESCRIPTION.
    R : (n-r,r) Array
        DESCRIPTION.
    zero_mat : (r,j-r) Array
        DESCRIPTION.
    F : (n-r,n-r) Array
        DESCRIPTION.
    FR : (n-r,r) Array
        DESCRIPTION.
    Ei_tau () Array
        average length of the game at the state i
    P_pow_n (n,n) Array
        compute P to the power n with the decompostion 
        tricks property from an absorb mat
    """
    

    # absorbing markov chain
    # if it's an absorbing chain so compute : -->
    if (absorbing == True):
        value = np.where(np.diag(mat) == 1)
        r = len(value[0])
        I = np.identity(r)
        zero_mat = np.zeros((r,n-r))
        # deleting row and columns in a same time
        Q = np.delete(np.delete(mat,[value],0),[value],1)
        R = np.delete(np.take(mat,[value],1),[value],0).reshape(n-r,r)
        if   (R == 0).all():
            R = np.delete(np.take(mat.T,[value],1),[value],0).reshape(n-r,r)
        new_mat = np.concatenate((np.concatenate((Q,R), axis=1),np.concatenate((zero_mat,I), axis=1)),axis=0)
        
        new_nu = np.delete(nu,[value[0]],1)
        "probabilité d'absorbtion:"
        " esperance du nombre de passage en j partant de i f_{i,j}"
        F = inv(np.identity(len(Q))-Q)
        
        "average length of the game at the state i is given by nu * sum F_i,j"
        Ei_tau =  sum( new_nu @ [F[i,:] for i in range(len(F))])
        " probability of being absorb in different state"
        FR = F@R

        P_pow_n = np.vstack((np.hstack((np.zeros(np.shape(Q)),FR)),np.hstack((zero_mat, I)))) 
        
        return I, Q, R, zero_mat, F, Ei_tau, FR, P_pow_n, new_mat
    else:
        P_pow_n = print("")
        print("not computable cause it's not an absorbing matrix")
        return P_pow_n
    
    
    
def asymptotic_probability(mat,n):
    """
    This algorithms compute the solution of  the asymptotic probability " means that : lim P(Xn=0)?"
    π_i @ mat = π_i under constraint sum for(i in n) π_i = 1
    
    we have the following system : π_i @ mat = π_i
    and we need to rewritte it to compute this : so the new problem will be written in the standar way:
        
        A = (P.Transpose() - Identity(n), and we add a row for the unknows values of the contraint : π_i
        A : (n+1,n)
        
        c = is full of zeros except for the last row which is equal to 1 and its the value of the sum of the constraint
        c : (n+1,1) Array

        we wanna solve the following system:
            A@π_i = z

    Parameters
    ----------
    mat : TYPE
        DESCRIPTION.
    

    Returns
    -------
    solution : (n,1) Array
        return the values of π_i 

    """
    
    print("\nThe invariant distribution of the chain is:")
    # implement the 
    c = np.append(np.zeros(n),1).reshape((-1,1))
    # mat - I_n + we add new row full of 1
    resol_mat = np.append(mat.T - np.identity(n) , np.ones(n).reshape((1,-1)), axis =0)

    #resol_mat = np.append(resol_mat, np.zeros(np.shape(resol_mat)[0]).reshape((-1,1)),axis=1)
    # we had to use lstsq cause we have more equation than unknows so it's compute
    # an approximation instead of inverse the matrix
    sol = lstsq(resol_mat,c)
    return sol 
        
if __name__ == '__main__':
    Matrix()
    Stochastic_matrix()
    type_mat()
    PVNu()
    transition_split()
    asymptotic_probability()

    
    



