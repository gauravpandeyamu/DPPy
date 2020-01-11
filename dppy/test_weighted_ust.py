import sys

from dppy.exotic_dpps import UST, UST_maze
from dppy.utils import find_all_spanning_trees
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from collections import Counter
from scipy.stats import chisquare



def sample_to_label(sample):
    """Join egdes of a sample to form the ID of the corresponding spanning tree"""
    return ''.join(map(str, sorted(map(sorted, sample.edges()))))
    
def edges_to_label(edges):
    return ''.join(map(str, sorted(map(sorted, edges))))



def plot_hist(G, tested_sampler=['Wilson', 'Aldous-Broder'], nbr_it=10000):
    """Plot an histogram checking the distribution of the spanning trees generated for weighted graph"""
    ust = UST(G)
    #ust.plot_graph()
    
    nbr_modes = len(tested_sampler)
    sample_seen = dict()
    
    for i in range(nbr_it):
        for k in range(nbr_modes):
            mode = tested_sampler[k]
            ust.sample(mode=mode)
            sample = ust.list_of_samples[-1]
            edges = np.array(sample.edges())
            sample_id = edges_to_label(edges)  
        
            if not sample_id in sample_seen:
                time_seen = np.zeros(nbr_modes)
                time_seen[k] = 1
                sample_seen[sample_id] = (time_seen, edges)
            else:
                sample_seen[sample_id][0][k] += 1
    
    labels = []
    values = [[] for k in range(nbr_modes)]
    
    list_items = sample_seen.items()
    for (id, (time_seen, edges)) in list_items:
        sample_weight = np.prod(np.array([G.get_edge_data(edges[i, 0], edges[i, 1])['weight'] for i in range(np.shape(edges)[0])]))
        labels.append(sample_weight)
        for k in range(nbr_modes):
            values[k].append(time_seen[k])
    n_labels = len(labels)
    labels = np.array(labels)
    index_sort = np.argsort(labels)
    labels = labels[index_sort]
    sorted_values = np.array([np.array(values[k])[index_sort] for k in range(nbr_modes)])
    sorted_values /= np.sum(sorted_values, axis=1)[:, None]
    
    #Add the theoretical distribution
    labels_sampler = tested_sampler[:]
    labels_sampler.append("Theoretical")
    nbr_modes += 1
    theo_values = labels / np.sum(labels)
    final_values = np.zeros((nbr_modes, n_labels))
    final_values[:-1, :] = sorted_values
    final_values[-1, :] = theo_values
    
    x_labels = np.arange(n_labels)
    x = np.zeros((nbr_modes, n_labels))
    width = 0.3
    for k in range(nbr_modes):
        x[k, :] = x_labels + (2*k - nbr_modes+1)*width/2
    
    fig, ax = plt.subplots()
    for k in range(nbr_modes):
        rects = ax.bar(x[k], final_values[k], width, label=labels_sampler[k])
        # # Uncomment to attach a text label above each bar displaying its height
        #for rect in rects:
        #    height = rect.get_height()
        #    ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                        
    ax.set_xlabel('Weight of the tree')
    ax.set_title('Check distribution of spanning tree generated after {} sampled of each procedure'.format(nbr_it))
    ax.set_xticks(x_labels)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()
    
    
    
def statistical_test(G, tested_sampler=['Wilson', 'Aldous-Broder'], nbr_it=10000, tol=0.05):
    """ Perform a chi-square test to check that the different spanning trees sampled have a distribution proportional to their weight"""
    ust = UST(G)
    all_spanning_trees = find_all_spanning_trees(G)
    nb_spanning_trees = len(all_spanning_trees)
    
    for mode in tested_sampler:
        ust.flush_samples()
        for _ in range(nbr_it):
            ust.sample(mode=mode)

        counter = Counter(map(sample_to_label, ust.list_of_samples))
        
        freq_tree_seen = np.array(list(counter.values())) / nbr_it
        n_seen = np.shape(freq_tree_seen)[0]
        freq = np.pad(freq_tree_seen, (0, nb_spanning_trees-n_seen), mode='constant', constant_values=0)
        theo = np.zeros(nb_spanning_trees)
        for i in range(nb_spanning_trees):
            edges = np.array(all_spanning_trees[i].edges())     
            sample_weight = np.prod(np.array([G.get_edge_data(edges[i, 0], edges[i, 1])['weight'] for i in range(np.shape(edges)[0])]))
            theo[i] = sample_weight
        theo /= np.sum(theo)
        
        _, pval = chisquare(f_obs=freq, f_exp=theo)
        
        if pval > tol:
            print("Test passed for mode " + mode, " ; p-value =", pval)
        else:
            print("Test not passed for mode " + mode, " ; p-value =", pval)
    

# Initialize graph
G = nx.Graph()
G.add_nodes_from(np.arange(5))
edges = [(0, 2, 1), (0, 3, 3), (1, 2, 5), (1, 4, 1), (2, 3, 1), (2, 4, 2), (3, 4, 2)]
G.add_weighted_edges_from(edges)
ust = UST(G)
ust.plot_graph()

#statistical_test(G, nbr_it=10000)

#plot_hist(G, nbr_it=10000)





    

