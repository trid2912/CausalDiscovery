import pandas as pd
import numpy as np
from bic import BaseEstimator, is_dag

data = pd.read_csv("Asia.csv")
data = data.drop('Unnamed: 0', axis=1)

base_estimate = BaseEstimator(data)

nodes = list(base_estimate.state_names.keys())
num_nodes = len(nodes)
print(nodes)

adj_matrix = np.zeros((num_nodes, num_nodes))
best_score = -999999
next_state = None
while True:
    for i in range(num_nodes):
        for j in range(num_nodes):
            copy_mat = adj_matrix.copy()
            copy_mat[i,j] = 1 - copy_mat[i,j]
            if (is_dag(copy_mat)):
                bic_score = 0
                for k in range(num_nodes):
                    parents = np.nonzero(copy_mat[:,k])[0]
                    parents = [nodes[k] for k in parents]
                    bic_score += base_estimate.local_score(nodes[k], parents)
                if bic_score > best_score:
                    next_state = copy_mat
                    best_score = bic_score
            else:
                continue
    print(next_state)
    print(best_score)
    if np.array_equal(next_state, adj_matrix):
        break
    adj_matrix = next_state

true_matrix = np.zeros((num_nodes, num_nodes))
true_matrix[0,2] = 1
true_matrix[1,3] = 1
true_matrix[1,4] = 1
true_matrix[2,5] = 1
true_matrix[3,5] = 1
true_matrix[4,7] = 1
true_matrix[5,6] = 1
true_matrix[5,7] = 1
print(is_dag(true_matrix))
bic_score = 0
for k in range(num_nodes):
    parents = np.nonzero(true_matrix[:,k])[0]
    parents = [nodes[k] for k in parents]
    bic_score += base_estimate.local_score(nodes[k], parents)
print(true_matrix)
print(bic_score)