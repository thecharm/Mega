import json
import math
import re
import sys
import time
from urllib.parse import quote

import networkx as nx
import matplotlib as plt
from nltk import Tree
from stanfordcorenlp import StanfordCoreNLP

from config import *


def getDependencyGraph(nlp, sentence, file_name, img_id, img_path):
    #print(sentence)
    dependency_tree = nlp.dependency_parse(sentence)
    #print(dependency_tree)
    words = nlp.word_tokenize(sentence)
    dependency_graph = {}
    words_len = len(words)
    f = open(file_name, 'a')
    for word in words:
        dependency_graph[word] = []
    for i in range(0, len(dependency_tree)):
        begin_idx = int(dependency_tree[i][1]) - 1
        end_idx = int(dependency_tree[i][2]) - 1
        if (begin_idx != -1):
            f.write('{} {}\n'.format(begin_idx, end_idx))

    # f.write('    ')
    with open(img_path, 'r', encoding='utf8') as f2:
        data_img = json.load((f2))
    if img_id not in data_img.keys():
        img_id = str(img_id) + '.jpg'
        if img_id not in data_img.keys():
            print(img_id)
    graph = data_img[img_id]['rel']
    for item in graph:
        f.write('{} {}\n'.format(item[0] + words_len, item[1] + words_len))


def getDependencyGraph_text_only(nlp, sentence, file_name):
    print(sentence)
    dependency_tree = nlp.dependency_parse(sentence)
    print(dependency_tree)
    words = nlp.word_tokenize(sentence)
    dependency_graph = {}
    words_len = len(words)
    f = open(file_name, 'a')
    for word in words:
        dependency_graph[word] = []
    for i in range(0, len(dependency_tree)):
        begin_idx = int(dependency_tree[i][1]) - 1
        end_idx = int(dependency_tree[i][2]) - 1
        if (begin_idx != -1):
            f.write('{} {}\n'.format(begin_idx, end_idx))


# Input: graph, RepMethod
# Output: dictionary of dictionaries: for each node, dictionary containing {node : {layer_num : [list of neighbors]}}
#        dictionary {node ID: degree}
def get_khop_neighbors(graph, rep_method):
    if rep_method.max_layer is None:
        rep_method.max_layer = graph.N  # Don't need this line, just sanity prevent infinite loop

    kneighbors_dict = {}

    # only 0-hop neighbor of a node is itself
    # neighbors of a node have nonzero connections to it in adj matrix
    for node in range(graph.N):
        neighbors = np.nonzero(graph.G_adj[node])[-1].tolist()  ###
        if len(neighbors) == 0:  # disconnected node
            print("Warning: node %d is disconnected" % node)
            kneighbors_dict[node] = {0: set([node]), 1: set()}
        else:
            if type(neighbors[0]) is list:
                neighbors = neighbors[0]
            kneighbors_dict[node] = {0: set([node]), 1: set(neighbors) - set([node])}
    print(kneighbors_dict)

    # For each node, keep track of neighbors we've already seen
    all_neighbors = {}
    for node in range(graph.N):
        all_neighbors[node] = set([node])
        all_neighbors[node] = all_neighbors[node].union(kneighbors_dict[node][1])
    print(all_neighbors)

    # Recursively compute neighbors in k
    # Neighbors of k-1 hop neighbors, unless we've already seen them before
    current_layer = 2  # need to at least consider neighbors
    while True:
        if rep_method.max_layer is not None and current_layer > rep_method.max_layer: break
        reached_max_layer = True  # whether we've reached the graph diameter
        print('graph.N', graph.N)
        for i in range(graph.N):
            # All neighbors k-1 hops away
            neighbors_prevhop = kneighbors_dict[i][current_layer - 1]

            khop_neighbors = set()
            # Add neighbors of each k-1 hop neighbors
            for n in neighbors_prevhop:
                neighbors_of_n = kneighbors_dict[n][1]
                for neighbor2nd in neighbors_of_n:
                    khop_neighbors.add(neighbor2nd)

            # Correction step: remove already seen nodes (k-hop neighbors reachable at shorter hop distance)
            khop_neighbors = khop_neighbors - all_neighbors[i]

            # Add neighbors at this hop to set of nodes we've already seen
            num_nodes_seen_before = len(all_neighbors[i])
            all_neighbors[i] = all_neighbors[i].union(khop_neighbors)
            num_nodes_seen_after = len(all_neighbors[i])

            # See if we've added any more neighbors
            # If so, we may not have reached the max layer: we have to see if these nodes have neighbors
            if len(khop_neighbors) > 0:
                reached_max_layer = False

            # add neighbors
            kneighbors_dict[i][current_layer] = khop_neighbors  # k-hop neighbors must be at least k hops away

        if reached_max_layer:
            break  # finished finding neighborhoods (to the depth that we want)
        else:
            current_layer += 1  # move out to next layer
        print(kneighbors_dict)
    return kneighbors_dict


# Turn lists of neighbors into a degree sequence
# Input: graph, RepMethod, node's neighbors at a given layer, the node
# Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_degree_sequence(graph, rep_method, kneighbors, current_node):
    if rep_method.num_buckets is not None:
        degree_counts = [0] * int(math.log(graph.max_degree, rep_method.num_buckets) + 1)
    else:
        degree_counts = [0] * (graph.max_degree + 1)

    # For each node in k-hop neighbors, count its degree
    for kn in kneighbors:
        weight = 1  # unweighted graphs supported here
        degree = graph.node_degrees[kn]
        if rep_method.num_buckets is not None:
            try:
                degree_counts[int(math.log(degree, rep_method.num_buckets))] += weight
            except:
                print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
        else:
            degree_counts[degree] += weight
    return degree_counts


# Get structural features for nodes in a graph based on degree sequences of neighbors
# Input: graph, RepMethod
# Output: nxD feature matrix
def get_features(graph, rep_method, verbose=True):
    print('graph.N:::', graph.N)
    before_khop = time.time()
    # Get k-hop neighbors of all nodes
    khop_neighbors_nobfs = get_khop_neighbors(graph, rep_method)

    graph.khop_neighbors = khop_neighbors_nobfs

    if verbose:
        print("max degree: ", graph.max_degree)
        after_khop = time.time()
        print("got k hop neighbors in time: ", after_khop - before_khop)

    G_adj = graph.G_adj
    num_nodes = G_adj.shape[0]
    if rep_method.num_buckets is None:  # 1 bin for every possible degree value
        num_features = graph.max_degree + 1  # count from 0 to max degree...could change if bucketizing degree sequences
    else:  # logarithmic binning with num_buckets as the base of logarithm (default: base 2)
        num_features = int(math.log(graph.max_degree, rep_method.num_buckets)) + 1
    feature_matrix = np.zeros((num_nodes, num_features))

    before_degseqs = time.time()
    for n in range(num_nodes):
        for layer in graph.khop_neighbors[
            n].keys():  # construct feature matrix one layer at a time
            if len(graph.khop_neighbors[n][layer]) > 0:
                # degree sequence of node n at layer "layer"
                deg_seq = get_degree_sequence(graph, rep_method, graph.khop_neighbors[n][layer], n)
                # add degree info from this degree sequence, weighted depending on layer and discount factor alpha
                feature_matrix[n] += [(rep_method.alpha ** layer) * x for x in deg_seq]
    after_degseqs = time.time()

    if verbose:
        print("got degree sequences in time: ", after_degseqs - before_degseqs)
    return feature_matrix


# Input: two vectors of the same length
# Optional: tuple of (same length) vectors of node attributes for corresponding nodes
# Output: number between 0 and 1 representing their similarity
def compute_similarity(graph, rep_method, vec1, vec2, node_attributes=None, node_indices=None):
    dist = rep_method.gammastruc * np.linalg.norm(vec1 - vec2)  # compare distances between structural identities
    if graph.node_attributes is not None:
        # distance is number of disagreeing attributes
        attr_dist = np.sum(graph.node_attributes[node_indices[0]] != graph.node_attributes[node_indices[1]])
        dist += rep_method.gammaattr * attr_dist
    return np.exp(-dist)  # convert distances (weighted by coefficients on structure and attributes) to similarities


# Sample landmark nodes (to compute all pairwise similarities to in Nystrom approx)
# Input: graph (just need graph size here), RepMethod (just need dimensionality here)
# Output: np array of node IDs
def get_sample_nodes(graph, rep_method, verbose=True):
    # Sample uniformly at random
    sample = np.random.permutation(np.arange(graph.N))[:rep_method.p]
    return sample


# Get dimensionality of learned representations
# Related to rank of similarity matrix approximations
# Input: graph, RepMethod
# Output: dimensionality of representations to learn (tied into rank of similarity matrix approximation)
def get_feature_dimensionality(graph, rep_method, verbose=True):
    p = int(rep_method.k * math.log(graph.N, 2))  # k*log(n) -- user can set k, default 10
    if verbose:
        print("feature dimensionality is ", min(p, graph.N))
    rep_method.p = min(p, graph.N)  # don't return larger dimensionality than # of nodes
    return rep_method.p


# xNetMF pipeline
def get_representations(graph, rep_method, verbose=True):
    # Node identity extraction
    # 获取图中第一部分右侧的内容
    feature_matrix = get_features(graph, rep_method, verbose)

    # Efficient similarity-based representation
    # Get landmark nodes
    if rep_method.p is None:
        rep_method.p = get_feature_dimensionality(graph, rep_method, verbose=verbose)  # k*log(n), where k = 10
    elif rep_method.p > graph.N:
        print("Warning: dimensionality greater than number of nodes. Reducing to n")
        rep_method.p = graph.N
    landmarks = get_sample_nodes(graph, rep_method, verbose=verbose)

    # Explicitly compute similarities of all nodes to these landmarks
    before_computesim = time.time()
    C = np.zeros((graph.N, rep_method.p))
    for node_index in range(graph.N):  # for each of N nodes
        for landmark_index in range(rep_method.p):  # for each of p landmarks
            # select the p-th landmark
            C[node_index, landmark_index] = compute_similarity(graph,
                                                               rep_method,
                                                               feature_matrix[node_index],
                                                               feature_matrix[landmarks[landmark_index]],
                                                               graph.node_attributes,
                                                               (node_index, landmarks[landmark_index]))

    before_computerep = time.time()

    # Compute Nystrom-based node embeddings
    W_pinv = np.linalg.pinv(C[landmarks])
    U, X, V = np.linalg.svd(W_pinv)
    Wfac = np.dot(U, np.diag(np.sqrt(X)))
    reprsn = np.dot(C, Wfac)
    after_computerep = time.time()
    if verbose:
        print("computed representation in time: ", after_computerep - before_computerep)

    # Post-processing step to normalize embeddings (true by default, for use with REGAL)
    if rep_method.normalize:
        reprsn = reprsn / np.linalg.norm(reprsn, axis=1).reshape((reprsn.shape[0], 1))
    return reprsn


if __name__ == "__main__":
    # nlp = StanfordCoreNLP(r'stanford-corenlp-4.2.0')
    # print(nlp.pos_tag('RT @NaraSeason : 0 market share : the downfall of Android and iOS competition'))
    # list = 'RT @NaraSeason : 0 market share : the downfall of Android and iOS competition'.split(' ')
    # for item in list:
    #     print(nlp.pos_tag(item)[0][1])

    # get the whole graph of text and image 
    nlp = StanfordCoreNLP(r'stanford-corenlp-4.2.0')
    kind="train"
    file_path = 'data_all_info/input/'+kind+'_data.json' # inf of dataset
    #file_path_save = 'data/text_graph_edges_20/' + kind + '/'# text only
    file_path_img = 'data/graph_of_image/'+kind+'.json' # scene graph inf

    with open(file_path, 'r', encoding='utf8') as f1:
        json_data1 = json.load(f1)
        file_path = 'data/combined_graph_edges_10/'+kind+'/'
        #file_path = 'data/text_graph_edges_5/' + kind + '/'#纯文本
        print(len(json_data1))
        #print(json_data1[1502]['img_id'])

        for i in range(0,len(json_data1)):
            sentence = json_data1[i]['text']
            img_id=json_data1[i]['img_id']

            sentence = sentence.replace('\n', '')
            sentence = sentence.strip()
            # remove url
            pattern = re.compile(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 匹配模式
            url = re.findall(pattern, sentence)
            # print(url)
            for item in url:
                sentence = sentence.replace(item, '')

            filename = file_path + str(i)
            getDependencyGraph(nlp,quote(sentence), filename,img_id,file_path_img)



