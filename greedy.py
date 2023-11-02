import pandas as pd
import numpy as np
from bic import BaseEstimator, is_dag

data = pd.read_csv("Asia.csv")
data = data.drop('Unnamed: 0', axis=1)

base_estimate = BaseEstimator(data)

nodes = list(base_estimate.state_names.keys())
num_nodes = len(nodes)

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
