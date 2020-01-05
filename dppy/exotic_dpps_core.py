# coding: utf8
""" Core functions for

- Uniform spanning trees

    * :func:`ust_sampler_wilson`
    * :func:`ust_sampler_aldous_broder`:

- Descent procresses :class:`Descent`:

    * :func:`uniform_permutation`

- :class:`PoissonizedPlancherel` measure

    * :func:`uniform_permutation`
    * :func:`RSK`: Robinson-Schensted-Knuth correspondande
    * :func:`xy_young_ru` young diagram -> russian convention coordinates
    * :func:`limit_shape`

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/exotic_dpps/index.html>`_
"""

import functools  # used for decorators to pass docstring

import numpy as np

# For Uniform Spanning Trees
import networkx as nx
from itertools import chain  # create graph edges from path

# For class PoissonizedPlancherel
from bisect import bisect_right  # for RSK

from dppy.utils import check_random_state


def ust_sampler_wilson(W, root=None,
                       random_state=None):
    """
    Compute a random spanning tree of a graph G given by his adjacency matrix
    
    param W:
        Adjacency matrix of the Graph
    param W:
        scipy.sparse.csr_matrix
    """
    rng = check_random_state(random_state)

    # Initialize the tree
    wilson_tree_graph = nx.Graph()
    #nb_nodes = len(list_of_neighbors)
    nb_nodes = W.shape[0]

    # Initialize the root, if root not specified start from any node
    n0 = root if root else rng.choice(nb_nodes)  # size=1)[0]
    # -1 = not visited / 0 = in path / 1 = in tree
    state = -np.ones(nb_nodes, dtype=int)
    state[n0] = 1
    nb_nodes_in_tree = 1

    path, branches = [], []  # branches of tree, temporary path

    while nb_nodes_in_tree < nb_nodes:  # |Tree| = |V| - 1

        # visit a neighbor of n0 uniformly at random
        #n1 = rng.choice(list_of_neighbors[n0])  # size=1)[0]
        weights = (W.getrow(n0).toarray())[0].astype('float')
        weights /= np.sum(weights)
        n1 = rng.choice(np.arange(nb_nodes), p=weights)

        if state[n1] == -1:  # not visited => continue the walk

            path.append(n1)  # add it to the path
            state[n1] = 0  # mark it as in the path
            n0 = n1  # continue the walk

        if state[n1] == 0:  # loop on the path => erase the loop

            knot = path.index(n1)  # find 1st appearence of n1 in the path
            nodes_loop = path[knot + 1:]  # identify nodes forming the loop
            del path[knot + 1:]  # erase the loop
            state[nodes_loop] = -1  # mark loopy nodes as not visited
            n0 = n1  # continue the walk

        elif state[n1] == 1:  # hits the tree => new branch

            if nb_nodes_in_tree == 1:
                branches.append([n1] + path)  # initial branch of the tree
            else:
                branches.append(path + [n1])  # path as a new branch

            state[path] = 1  # mark nodes in path as in the tree
            nb_nodes_in_tree += len(path)

            # Restart the walk from a random node among those not visited
            nodes_not_visited = np.where(state == -1)[0]
            if nodes_not_visited.size:
                n0 = rng.choice(nodes_not_visited)  # size=1)[0]
                path = [n0]

    tree_edges = list(chain.from_iterable(map(lambda x: zip(x[:-1], x[1:]),
                                              branches)))
    wilson_tree_graph.add_edges_from(tree_edges)

    return wilson_tree_graph


def ust_sampler_wilson_nodes(W, absorbing_weight=0, random_state=None):
    """
    Implement the Wilson's algorthm described in Graph sampling with determinantal processes, Nicolas Tremblay et al., 2017
    It samples a set of nodes from the directed graph of adjacency matrix W and extracts a random spanning tree from the trajectories of the random walks performed.
    
    
    :param W:
        Adjacency matrix of the graph
    :type W:
        scipy.sparse.csr_matrix

    :param absorbing_weight:
        Weight of the node Delta added to the graph
    :type absorbing_weight:
        int


    :return Y:
        Set of nodes selected by the algorithm
    :rtype Y:
        list
        
    :return all_path:
        All the trajectories of the random walks performed on the augmented graph
    :rtype all_path:
        list of list
        
    :return wilson_tree_from_path:
        Random spanning tree of the graph
    :rtype wilson_tree_from_path:
        network Graph
    """
        
    rng = check_random_state(random_state)

    # Initialization
    nb_nodes = W.shape[0]
    Y = []
    Nu = np.zeros(nb_nodes, dtype=bool)
    all_path = []
    
    # Compute the probabilities of transition
    transition_probabilities = np.pad(W.toarray(), [(0, 0), (0, 1)], mode='constant', constant_values=absorbing_weight).astype('float')
    norm = np.sum(transition_probabilities, axis=1)
    transition_probabilities[np.nonzero(norm), :] /= norm[np.nonzero(norm), None]
    
    while np.sum(Nu) != nb_nodes:
        
        # Initialize the root,
        walk_index = rng.choice(np.nonzero(np.invert(Nu))[0])
        
        visited_nodes = np.zeros(nb_nodes, dtype=bool)
        visited_nodes[walk_index] = True
        path = [walk_index]
        print("start path", walk_index)
            
        while True:
            
            transition = transition_probabilities[walk_index]
            
            if absorbing_weight == 0 and np.sum(transition) == 0:
                print("Case 0")
                Nu[visited_nodes] = True
                all_path.append(path)
                break
            
            # Get next node
            next_index = rng.choice(nb_nodes+1, p=transition)
            
            # If we end up in the sink, add node to Y, add path to Nu and quit
            if next_index == nb_nodes:
                print("Case 1")
                Nu[visited_nodes] = 1
                Y.append(walk_index)
                all_path.append(path)
                break
            
            # If we end in a node in Nu, add path to Nu and quit
            elif Nu[next_index]:
                print("Case 2")
                Nu[visited_nodes] = 1
                path.append(next_index)
                all_path.append(path)
                break
                
            # If we loop over ourselves, erase the entire loop
            elif visited_nodes[next_index]:
                print("Case 3")
                if sum(visited_nodes) == nb_nodes:
                    print('Case 3a')
                    Nu[visited_nodes] = True
                    all_path.append(path)
                    break
                first_appearence = path.index(next_index)  # find 1st appearence of next_index in the path
                nodes_loop = path[first_appearence:]  # identify nodes forming the loop
                del path[first_appearence:]  # erase the loop
                visited_nodes[nodes_loop] = False  # mark loopy nodes as not visited
                
                
            # Else continue walk
            walk_index = next_index
            visited_nodes[next_index] = True
            path.append(next_index)
            #print("Nu=", Nu)
            print("path=", path)
            
    wilson_tree_from_path = nx.Graph()
    print("all_path=", all_path)
    tree_edges = list(chain.from_iterable(map(lambda x: zip(x[:-1], x[1:]), all_path)))
    print("tree_edges=", tree_edges)
    wilson_tree_from_path.add_edges_from(tree_edges)
    
    #print("Nu=", Nu)
    return Y, all_path, wilson_tree_from_path

def ust_sampler_aldous_broder(W, root=None,
                              random_state=None):

    rng = check_random_state(random_state)

    # Initialize the tree
    aldous_tree_graph = nx.Graph()
    #nb_nodes = len(list_of_neighbors)
    nb_nodes = W.shape[0]

    # Initialize the root, if root not specified start from any node
    n0 = root if root else rng.choice(nb_nodes)  # size=1)[0]
    visited = np.zeros(nb_nodes, dtype=bool)
    visited[n0] = True
    nb_nodes_in_tree = 1

    tree_edges = np.zeros((nb_nodes - 1, 2), dtype=np.int)

    while nb_nodes_in_tree < nb_nodes:

        # visit a neighbor of n0 uniformly at random
        #n1 = rng.choice(list_of_neighbors[n0])  # size=1)[0]
        weights = (W.getrow(n0).toarray())[0].astype('float')
        weights /= np.sum(weights)
        n1 = rng.choice(np.arange(nb_nodes), p=weights)

        if visited[n1]:
            pass  # continue the walk
        else:  # create edge (n0, n1) and continue the walk
            tree_edges[nb_nodes_in_tree - 1] = [n0, n1]
            visited[n1] = True  # mark it as in the tree
            nb_nodes_in_tree += 1

        n0 = n1

    aldous_tree_graph.add_edges_from(tree_edges)

    return aldous_tree_graph


def uniform_permutation(N, random_state=None):
    """ Draw a perputation :math:`\\sigma \\in \\mathfrak{S}_N` uniformly at random using Fisher-Yates' algorithm

    .. seealso::

        - `Fisher–Yates_shuffle <https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle>_

        - `Numpy shuffle <https://github.com/numpy/numpy/blob/d429f0fe16c0407509b1f20d997bf94f1027f61b/numpy/random/mtrand.pyx#L4027>_`
    """
    rng = check_random_state(random_state)

    sigma = np.arange(N)
    for i in range(N - 1, 0, -1):  # reversed(range(1, N))
        j = rng.randint(0, i + 1)
        if j == i:
            continue
        sigma[j], sigma[i] = sigma[i], sigma[j]

    # for i in range(N - 1):
    #     j = rng.randint(i, N)
    #     sigma[j], sigma[i] = sigma[i], sigma[j]

    return sigma


def RSK(sequence):
    """Apply Robinson-Schensted-Knuth correspondence on a sequence of reals, e.g. a permutation, and return the corresponding insertion and recording tableaux.

    :param sequence:
        Sequence of real numbers
    :type sequence:
        array_like

    :return:
        :math:`P, Q` insertion and recording tableaux
    :rtype:
        list

    .. seealso::

        `RSK Wikipedia <https://en.wikipedia.org/wiki/Robinson%E2%80%93Schensted%E2%80%93Knuth_correspondence>`_
    """

    P, Q = [], []  # Insertion/Recording tableau

    for it, x in enumerate(sequence, start=1):

        # Iterate along the rows of the tableau P to find a place for the bouncing x and record the position where it is inserted
        for row_P, row_Q in zip(P, Q):

            # If x finds a place at the end of a row of P
            if x >= row_P[-1]:
                row_P.append(x)  # add the element at the end of the row of P
                row_Q.append(it)  # record its position in the row of Q
                break
            else:
                # find place for x in the row of P to keep the row ordered
                ind_insert = bisect_right(row_P, x)
                # Swap x with the value in place
                x, row_P[ind_insert] = row_P[ind_insert], x

        # If no room for x at the end of any row of P create a new row
        else:
            P.append([x])
            Q.append([it])

    return P, Q


def xy_young_ru(young_diag):
    """ Compute the xy coordinates of the boxes defining the young diagram, using the russian convention.

    :param young_diag:
        points
    :type  young_diag:
        array_like

    :return:
        :math:`\\omega(x)`
    :rtype:
        array_like
    """

    def intertwine(arr_1, arr_2):
        inter = np.empty((arr_1.size + arr_2.size,), dtype=arr_1.dtype)
        inter[0::2], inter[1::2] = arr_1, arr_2
        return inter

    # horizontal lines
    x_hor = intertwine(np.zeros_like(young_diag), young_diag)
    y_hor = np.repeat(np.arange(1, young_diag.size + 1), repeats=2)

    # vertical lines
    uniq, ind = np.unique(young_diag[::-1], return_index=True)
    gaps = np.ediff1d(uniq, to_begin=young_diag[-1])

    x_vert = np.repeat(np.arange(1, 1 + gaps.sum()), repeats=2)
    y_vert = np.repeat(young_diag.size - ind, repeats=gaps)
    y_vert = intertwine(np.zeros_like(y_vert), y_vert)

    xy_young_fr = np.column_stack(
        [np.hstack([x_hor, x_vert]), np.hstack([y_hor, y_vert])])

    rot_45_and_scale = np.array([[1.0, -1.0],
                                 [1.0, 1.0]])

    return xy_young_fr.dot(rot_45_and_scale.T)


def limit_shape(x):
    """ Evaluate :math:`\\omega(x)` the limit-shape function :cite:`Ker96`

    .. math::

        \\omega(x) =
        \\begin{cases}
            |x|, &\\text{if } |x|\\geq 2\\
            \\frac{2}{\\pi} \\left(x \\arcsin\\left(\\frac{x}{2}\\right) + \\sqrt{4-x^2} \\right) &\\text{otherwise } \\end{cases}

    :param x:
        points
    :type x:
        array_like

    :return:
        :math:`\\omega(x)`
    :rtype:
        array_like

    .. seealso::

        - :func:`plot_diagram <plot_diagram>`
        - :cite:`Ker96`
    """

    w_x = np.zeros_like(x)

    abs_x_gt2 = np.abs(x) >= 2.0

    w_x[abs_x_gt2] = np.abs(x[abs_x_gt2])
    w_x[~abs_x_gt2] = x[~abs_x_gt2] * np.arcsin(0.5 * x[~abs_x_gt2])\
                      + np.sqrt(4.0 - x[~abs_x_gt2]**2)
    w_x[~abs_x_gt2] *= 2.0 / np.pi

    return w_x
