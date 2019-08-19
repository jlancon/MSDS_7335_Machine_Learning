# -*- coding: utf-8 -*-
"""

Created on Wed July 10 18:00:00 2019

@author: Yejur Kunwar
*****************************************************************************************
@credit: 1. Chris for his base codes and instructions

         2. Documentations from (https://scikit-learn.org/stable/)

         3. In Depth: Principal Component Analysis (https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)

         5. Discussion and collaboration with David Stroud and Jefferey Lancon.

*****************************************************************************************

# Decision making with Matrices

# This is a pretty simple assignment.  You will do something you do everyday, but today it will be with matrix manipulations.

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.
people = {'Jane': {'willingness to travel':
                  'desire for new experience':
                  'cost':
                  'indian food':
                  'mexican food':
                  'hipster points':
                  'vegitarian':
                  }

          }

# Transform the user data into a matrix(M_people). Keep track of column and row ids.



# Next you collected data from an internet website. You got the following information.

resturants  = {'flacos':{'distance' :
                        'novelty' :
                        'cost':
                        'average rating':
                        'cuisine':
                        'vegitarians'
                        }

}


# Transform the restaurant data into a matrix(M_resturants) use the same column index.

# The most imporant idea in this project is the idea of a linear combination.

# 1. Informally describe what a linear combination is  and how it will relate to our resturant matrix.

# 2. Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.

# 3. Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?

# 4. Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

# 5. Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.

# 6. Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# 7. How should you preprocess your data to remove this problem.

# 8. Find  user profiles that are problematic, explain why?

# 9. Think of two metrics to compute the disatistifaction with the group.

# 10. Should you split in two groups today?

# 11. Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

# 12. Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?

*****************************************************************************************


"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import matplotlib.cm as cm  #https://matplotlib.org/api/cm_api.html
from collections import OrderedDict
import json

class manipulationDecision(object):
    """Lets the matrx manupulation on making decision using linear combination method."""
    def __init__(self):
        """You asked your 10 work friends to answer a survey. They gave you back the following dictionary object."""

        self.people = {'Jane': {'willingness to travel': 0.1596993,
                          'desire for new experience':0.67131344,
                          'cost':0.15006726,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.01892123,
                          },
                  'Bob': {'willingness to travel': 0.63124581,
                          'desire for new experience':0.20269888,
                          'cost':0.01354308,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.15251223,
                          },
                  'Mary': {'willingness to travel': 0.49337138 ,
                          'desire for new experience': 0.41879654,
                          'cost': 0.05525843,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.03257365,
                          },
                  'Mike': {'willingness to travel': 0.08936756,
                          'desire for new experience': 0.14813813,
                          'cost': 0.43602425,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.32647006,
                          },
                  'Alice': {'willingness to travel': 0.05846052,
                          'desire for new experience': 0.6550466,
                          'cost': 0.1020457,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.18444717,
                          },
                  'Skip': {'willingness to travel': 0.08534087,
                          'desire for new experience': 0.20286902,
                          'cost': 0.49978215,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.21200796,
                          },
                  'Kira': {'willingness to travel': 0.14621567,
                          'desire for new experience': 0.08325185,
                          'cost': 0.59864525,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.17188723,
                          },
                  'Moe': {'willingness to travel': 0.05101531,
                          'desire for new experience': 0.03976796,
                          'cost': 0.06372092,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.84549581,
                          },
                  'Sara': {'willingness to travel': 0.18780828,
                          'desire for new experience': 0.59094026,
                          'cost': 0.08490399,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.13634747,
                          },
                  'Tom': {'willingness to travel': 0.77606127,
                          'desire for new experience': 0.06586204,
                          'cost': 0.14484121,
                          #'indian food':1,
                          #'Mexican food':1,
                          #'hipster points':3,
                          'vegetarian': 0.01323548,
                          }
                  }

        # Transform the user data into a matrix(M_people). Keep track of column and row ids.

                                               # convert each person's values to a list

        peopleKeys, peopleValues = [], []
        lastKey = 0
        for k1, v1 in self.people.items():
            row = []

            for k2, v2 in v1.items():
                peopleKeys.append(k1+'_'+k2)
                if k1 == lastKey:
                    row.append(v2)
                    lastKey = k1

                else:
                    peopleValues.append(row)
                    row.append(v2)
                    lastKey = k1


        #here are some lists that show column keys and values
#         print(peopleKeys)
#         print(peopleValues)



        self.peopleMatrix = np.array(peopleValues)

#         peopleMatrix.shape


        # Next you collected data from an internet website. You got the following information.

        #1 is bad, 5 is great

        self.restaurants  = {'flacos':{'distance' : 2,
                                'novelty' : 3,
                                'cost': 4,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 5
                                },
                      'Joes':{'distance' : 5,
                                'novelty' : 1,
                                'cost': 5,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 3
                              },
                      'Poke':{'distance' : 4,
                                'novelty' : 2,
                                'cost': 4,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 4
                              },
                      'Sush-shi':{'distance' : 4,
                                'novelty' : 3,
                                'cost': 4,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 4
                              },
                      'Chick Fillet':{'distance' : 3,
                                'novelty' : 2,
                                'cost': 5,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 5
                              },
                      'Mackie Des':{'distance' : 2,
                                'novelty' : 3,
                                'cost': 4,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 3
                              },
                      'Michaels':{'distance' : 2,
                                'novelty' : 1,
                                'cost': 1,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 5
                              },
                      'Amaze':{'distance' : 3,
                                'novelty' : 5,
                                'cost': 2,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 4
                              },
                      'Kappa':{'distance' : 5,
                                'novelty' : 1,
                                'cost': 2,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 3
                              },
                      'Mu':{'distance' : 3,
                                'novelty' : 1,
                                'cost': 5,
                                #'average rating': 5,
                                #'cuisine': 5,
                                'vegetarian': 3
                              }
        }


        # Transform the restaurant data into a matrix(M_resturants) use the same column index.


        restaurantsKeys, restaurantsValues = [], []

        for k1, v1 in self.restaurants.items():
            for k2, v2 in v1.items():
                restaurantsKeys.append(k1+'_'+k2)
                restaurantsValues.append(v2)

        #here are some lists that show column keys and values
#         print(restaurantsKeys)
#         print(restaurantsValues)

#         len(restaurantsValues)
        #reshape to 2 rows and 4 columns

        #converting lists to np.arrays is easy
        self.restaurantsMatrix = np.reshape(restaurantsValues, (10,4))

# Choose a person and compute(using a linear combination) the top restaurant for them.
# What does each entry in the resulting vector represent?

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.
# What does the a_ij matrix represent?

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
# I believe that this is what John and  is asking for, sum by columns


    def score_calc(self):
        newPeopleMatrix = np.swapaxes(result.peopleMatrix, 1,0)
        all_score = np.matmul(self.restaurantsMatrix, newPeopleMatrix)
        print("A new Matrix from all people, \nRows are Restaurants, Columns are People")
        return all_score

    def optimal_score(self, score):
        reskey = np.array(list(result.restaurants.keys()))
        each = np.sum(score, axis=1)
        overall_resscore = map(lambda x, y: str(x) + " : " + str(y), reskey, each)
        for i in list(overall_resscore):
            print('\t',i)

    def rank_calc(self, score):
        sortedResults = score.argsort()[::-1]
        np.sum(sortedResults, axis=1)
        temp = score.argsort()
        ranks = np.arange(len(score))[temp.argsort()]+1
        return ranks

    def heatmap(self, score):
        plot_dims = (12,10)
        fig, ax = plt.subplots(figsize=plot_dims)
        sns.heatmap(ax=ax, data=score, annot=True)
        plt.savefig('Heatmap_X_People_Y_Restaurant.png')
        plt.show()

#This function was shamefully taken from the below and modified for our purposes
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# plot principal components

    def perform_pca(self,matrix):
        pca = PCA(n_components=2)
        MatrixPcaTransform = pca.fit_transform(result.peopleMatrix)
        print("PCA Components: \n",pca.components_)
        print("PCA explained variances : \n",pca.explained_variance_)
        return MatrixPcaTransform

    def heirarchy_cluster(self,matrix, labellist, name):
        linked = linkage(matrix, 'single')
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)
        dendrogram(linked,
                orientation='top',
                labels=labellist,
                distance_sort='descending',
                show_leaf_counts=True, ax=ax)
        ax.tick_params(axis='x', which='major', labelsize=25)
        ax.tick_params(axis='y', which='major', labelsize=25)
        plt.savefig('Hierarchial Cluster '+ str(name) + '.png')
        plt.show()



if __name__ == "__main__":
    result = manipulationDecision()
    print("DECISION MAKING WITH MATRICES")
    print("**"*50)
    print("Transform the user data into a matrix(M_people). Keep track of column and row ids.\n")
    print(result.people.keys())
    print(result.peopleMatrix)
    print("**"*50)
    print("Transform the restaurant data into a matrix(M_resturants) use the same column index.\n")
    print(result.restaurants.keys())
    print(result.restaurantsMatrix)
    print("**"*50)
    print("Informally describe what a linear combination is  and how it will relate to our resturant matrix.")
    print("\t The linear combination is method where two or more components with N dimensions are multiplied by constanst to see interaction.")
    print("**"*50)
    score = result.score_calc()
    print(score)
    print("**"*50)
    print("Optimal Score restaurant for all users")
    result.optimal_score(score)
    ran = result.rank_calc(score)
    print(ran)
    print("**"*50)
    print("""Why is there a difference between the two?  What problem arrives?  What does represent in the real world? How should you preprocess your data to remove this problem. Find  user profiles that are problematic, explain why? Think of two metrics to compute the disatistifaction with the group. Should you split in two groups today? Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?""")
    print("\t Summing the ranks according to score using first method results in array that is equal accross all restaurant which does not work. In real world this is a problem because it forces one person to follow other with out preference. If we apply weights on this scale then it may solve problem, still someone will have to compromise. We can do one of several things i.e. PCA or Kmeans clustering to find set of people that have comparable choices.")
    result.heatmap(score)
    print("X axis is :", list(result.people.keys()), "\n0-->9")
    print("Y axis is :", list(result.restaurants.keys()), "\n0-->9")
    print("**"*50)
    print("People --> PCA and Clustering ")
    peopleMatrixPcaTransform = result.perform_pca(result.peopleMatrix)
    print("People PCA Matrix: \n",peopleMatrixPcaTransform)
    print("**"*50)
    print("Restaurant --> PCA and Clustering ")
    restaurantsMatrixPcaTransform = result.perform_pca(result.restaurantsMatrix)
    print("Restaurant PCA Matrix: \n",restaurantsMatrixPcaTransform)
    peoplelabelList = ['Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom']
    print("**"*50)
    print("People Hierarchial Clustering")
    result.heirarchy_cluster(peopleMatrixPcaTransform,peoplelabelList, "People")
    print("**"*50)
    print("Restaurant Hierarchial Clustering")
    restauranrlabelList = ['Flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu']
    result.heirarchy_cluster(restaurantsMatrixPcaTransform,restauranrlabelList, "Restaurant")
    print("**"*50)
    print('From Cluster, \n')
    print("Group 0 is Mike, Skip, Kira, and Moe")
    group0 = ran[0:,[3,5,6,7]]
    print(np.sum(group0, axis=1))
    print("\tGroup 0 wants to go to flacos or Chick Fillet (it is a tie)")
    print("\t-"*10)
    print("Group 1 is Bob, Mary, and Tom")
    group1 = ran[0:,[1,2,9]]
    print(np.sum(group1, axis=1))
    print("\tGroup 1 wants to go to Joes")
    print("\t-"*10)
    print("Group 2 is Jane, Alice, and Sara")
    group2 = ran[0:,[0,4,8]]
    print(np.sum(group2, axis=1))
    print("\tGroup 2 wants to go to Amaze")
    print("**"*50)
    print("""\nTommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.Can you find their weight matrix?""")
    print("""\t If we know restaurantsMatrix and Results Matrix can we calculate peopleMatrix? If the matrix A is invertible, then the equation Ax=b has a unique solution, namely x=A−1b.If A is not invertible, there may be either zero or many solutions to your problem.""")
    b = score
    ainv = np.linalg.pinv(result.restaurantsMatrix)
    x = np.matmul(ainv,b)
    print(x)
    print(np.allclose(result.peopleMatrix, x.T, rtol=1e-14, atol=1e-14, equal_nan=False))
    print("""\tThe problem is going from the rank matrix to the results matrix.If you had the results matrix, finding the people weights matrix with the pinv of the Restaurant matrix would be trivial.
    However, we don't have the results matrix, we have a ranking of the results matrix. The ranking matrix causes a loss of information from the results matrix, which may not be recoverable.
    With the rankings matrix however, if you could find the clusters each person belongs to, you could still come to a conclusion about the place(s) to take the other team to lunch.""")
    print("**"*50)
    print("END OF ANALYSIS")
