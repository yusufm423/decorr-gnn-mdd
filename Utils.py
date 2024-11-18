import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data

def fisher_z_transform(correlation_matrix, epsilon=1e-5):
    return 0.5 * np.log((1 + correlation_matrix) / (1 - correlation_matrix + epsilon))

def create_graph(X, Y, start, end, region=True):
    X_adjgraph=[]
    X_featgraph = []

    for i in range(len(Y)):
        if region == True:
            bold_matrix = X[i] #RxR ........?
        else:
            bold_matrix = np.transpose(X[i][start:end,:]) #TxT
        
        window_data1 = np.corrcoef(bold_matrix)
        correlation_matrix_fisher = fisher_z_transform(window_data1)
        correlation_matrix_fisher = np.around(correlation_matrix_fisher, 8)
        knn_graph = compute_KNN_graph(correlation_matrix_fisher)

        if region == True:
            X_featgraph.append(correlation_matrix_fisher)
        else:
            X_featgraph.append(bold_matrix)
            
        X_adjgraph.append(knn_graph)

    return X_featgraph, X_adjgraph, Y



def create_graph_ml(X, Y, start, end, region=True):
    X_adjgraph=[]
    X_featgraph = []

    for i in range(len(Y)):
        if region == True:
            bold_matrix = X[i][start:end,:] #RxR ................?
        else:
            bold_matrix = np.transpose(X[i][start:end,:]) #TxT
        
        window_data1 = np.corrcoef(bold_matrix)
        window_data1[window_data1 < 0.8] = 0
        window_data1[window_data1 >= 0.8 ] = 1
        #column_means = np.mean(window_data1, axis = 0)
        #correlation_matrix_fisher = fisher_z_transform(window_data1)
        correlation_matrix = np.around(window_data1, 8)
        knn_graph = compute_KNN_graph(correlation_matrix)

        if region == True:
            X_featgraph.append(window_data1)
        else:
            X_featgraph.append(bold_matrix)
            
        X_adjgraph.append(knn_graph)

    return X_featgraph, X_adjgraph, Y




def to_tensor(X_featgraph, X_adjgraph, Y):
    datalist = []
    
    for i in range(len(Y)):
        ty = Y[i]

        y = torch.tensor([ty]).long()

        adjacency = X_adjgraph[i]
        feature = X_featgraph[i]

        x = torch.from_numpy(feature).float()
        adj= adjacency
        adj = torch.from_numpy(adj).float()
        edge_index, edge_attr = dense_to_sparse(adj)
        
        datalist.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return datalist


#sliding window
def create_graph_sliding_window(X, Y, start, end, region=True):
    S = 30 # Sliding Step
    T = 60 # Window Size

    X_adjgraph=[]
    X_featgraph = []
    Y_list = []
    num_samples_per_subject = []

    for i in range(len(Y)):
        #select rows according to atlas
        bold_matrix = X[i] #RxR..............?
        temp_y = Y[i]

        num_rows, num_cols = bold_matrix.shape
        num_samples = 0
        for start_idx in range(0, num_cols - T + 1, S):
            end_idx = start_idx + T
            if end_idx <= num_cols:
                
                if region == True:
                    window_data =  bold_matrix[:, start_idx:end_idx]    #RxR
                else:
                    window_data =  np.transpose(bold_matrix[:, start_idx:end_idx])  #TxT

                window_data1 = np.corrcoef(window_data)
                correlation_matrix_fisher = fisher_z_transform(window_data1)
                correlation_matrix_fisher = np.around(correlation_matrix_fisher, 8)     #upto 8 decimal points
                knn_graph = compute_KNN_graph(correlation_matrix_fisher)

                if region == True:
                    X_featgraph.append(correlation_matrix_fisher)
                else:
                    X_featgraph.append(window_data)
                    
                X_adjgraph.append(knn_graph)
                Y_list.append(temp_y)
                num_samples = num_samples+1
            
        num_samples_per_subject.append(num_samples)


    return X_featgraph, X_adjgraph, Y_list, num_samples_per_subject

def compute_KNN_graph(matrix, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)

    return A


def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()




def create_graph_freq(X,Y):
    X_adjgraph=[]
    
    for i in range(len(Y)):
        correlation_matrix_fisher = fisher_z_transform(X)
        correlation_matrix_fisher = np.around(correlation_matrix_fisher,4)
        knn_graph = compute_KNN_graph(correlation_matrix_fisher)

    return knn_graph
