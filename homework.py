from main_empirical import *
from main_theory import *
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(198768)

nu_iter=800

##################################################################
##################################################################
"""-----------------A is an absorbing matrices-----------------"""
##################################################################
##################################################################
x=1/7

A =  np.array([[0,1,0,0,0],
              [(1-x),0,x,0,0],
              [0,(1-x),0,x,0],
              [0,0,(1-x),0,x],
              [0,0,0,0,1]])


#nu = np.array([[1,0,0,0,0]])
nu = np.array([[1/10,2/10,4/10,2/10,1/10]])


n = 7000
k=8
"""------------theoritical values------------"""
mat_sto = Stochastic_matrix(A, len(A))
absor, regular, irreducible = type_mat(A,len(A))
print("Absorbing =" ,absor,", Regular =", regular, ", Irreducible =",irreducible)

if absor== True: 
        PVN = PVNu(A,len(A),nu,k, absorbing = True)
        I, Q, R, zero_mat, F,Ei_tau, FR, P_pow_n, new_matA = transition_split(len(A),A,nu, absor)
        print("\n I = \n",I, "\n\n Q = \n" ,Q,"\n\n R = \n" ,R,"\n\n zero_mat = \n", zero_mat,"\n\n F = \n", F,
              "\n\n FR = \n", FR,"\n\n P power n = \n", P_pow_n)
        print(" \naverage length of the game at state i is given by sum nu_i * F_i,j: \n Ei_tau = \n",  Ei_tau)
        
else: 
        P_pow_n = transition_split(len(A),A,nu, absor)
        print("\n P power n = \n", P_pow_n)
        PVN = PVNu(A,len(A),nu,k)      

if (regular == True and irreducible== True):
        sol = asymptotic_probability(A,len(A))
        print("\n The solution to the problem π_i @ P = π_i is : \n π_i = \n it's also the lim_{n->+∞}P_1(X_n=i) = π_i = \n",sol[0])
        print("the average return times are given by E_i(Tau_i) = \n", 1/sol[0])




"""-------------Empirical values-----------"""

XA = markov_chain(n,A,nu,nu_iter)
timesA, mean,yA, lstA = statistique_markov_chain(n, A, nu_iter, XA)


print("\n Markov chain A time to be in an absorbing for the first simulation is: ", lstA[0], "\nWith respect to the initial distribution", nu)
print("\n For the simulation of :", nu_iter,"iteration of Markov chain, the average time before absorption is:",Average(lstA))
print('\n Average times in a state :',timesA.mean(axis=0), "for a length of the markov chain:",n,"and :",nu_iter,"simulation of ramdom walk" )


#iteration = int(input("\n Number of graph that you want must be lower than \n  the number of nu_iter : "))
iteration = 2
for j in range(iteration):
   fig, ax = plt.subplots()
   for i in range(len(A)):
       ax.plot(range(n), yA[:,j,i], label = i)
       ax.legend(loc = 'upper right')
       ax.set_title('Probability to be in a states i over times')



##################################################################
##################################################################
"""----------------B is an irreducible matrices----------------"""
##################################################################
##################################################################
nu_iter=500
B = np.array([[0.7, 0.1, 0.2],
               [0.2, 0.1, 0.7],
               [0.7, 0.2, 0.1]])
nu1 = np.array([[0.2,0.3,0.5]])
n1= 2500

"""------------------------"""
mat_sto = Stochastic_matrix(B,len(B))
absor, regular, irreducible = type_mat(B,len(B))
print("Absorbing =" ,absor,", Regular =", regular, ", Irreducible =",irreducible)
if absor== True: 
        PVN = PVNu(B,len(B),nu1,k, absorbing = True)
        I, Q, R, zero_mat, F,Ei_tau, FR, P_pow_n, new_mat = transition_split(len(B),B,nu1, absor)
        print("\n I = \n",I, "\n\n Q = \n" ,Q,"\n\n R = \n" ,R,"\n\n zero_mat = \n", zero_mat,"\n\n F = \n", F,
              "\n\n FR = \n", FR,"\n\n P power n = \n", P_pow_n)
        print(" \naverage length of the game at state i is given by sum nu_i * F_i,j: \n Ei_tau = \n",  Ei_tau)
        
else: 
        P_pow_n = transition_split(len(B),B,nu1, absor)
        print("\n P power n = \n", P_pow_n)
        PVN = PVNu(B,len(B),nu1,k)      

if (regular == True and irreducible== True):
        sol = asymptotic_probability(B,len(B))
        print("\n The solution to the problem π_i @ P = π_i is : \n π_i = \n it's also the lim_{n->+∞}P_1(X_n=i) = π_i = \n",sol[0])
        print("the average return times are given by E_i(Tau_i) = \n", 1/sol[0])



"""-------------Empirical values-----------"""

XB = markov_chain(n1,B,nu1,nu_iter)
timesB, meanB,yB,lstB = statistique_markov_chain(n1, B, nu_iter, XB)

# average return of an irreducible matrices is N/times
print("\n Times spent in the matrices B for the 1st simulation in a state is : ", timesB[1,:])
print("\n Average times spent in the matrices B for", nu_iter, "simulation is :", timesB.mean(axis=0))
print("\n Average probability times spent in the matrices B for", nu_iter,"and",n1, "step in the simulation is :", timesB.mean(axis=0)/n1)
print("\n Average return times in the matrices B in state i is : \n" , n1/timesB.mean(axis=0))


iteration = 2
for j in range(iteration):
   fig, ax = plt.subplots()
   for i in range(len(B)):
       ax.plot(range(n1), yB[:,j,i], label = i)
       ax.legend(loc = 'upper right')
       ax.set_title('Probability to be in a states i over times')



