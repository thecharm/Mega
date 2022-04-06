import numpy as np
import scipy.io as sio
import sklearn.metrics.pairwise
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
import scipy.sparse as sp
from scipy.spatial.distance import cosine


def get_embedding_similarities(embed, embed2=None, sim_measure="euclidean", num_top=None):
    n_nodes, dim = embed.shape
    if embed2 is None:
        embed2 = embed

    if num_top is not None:  # KD tree with only top similarities computed
        kd_sim = kd_align(embed, embed2, distance_metric=sim_measure, num_top=num_top)
        return kd_sim

    # All pairwise distance computation
    if sim_measure == "cosine":
        similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(embed, embed2)
    else:
        similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed, embed2)
        similarity_matrix = np.exp(-similarity_matrix)

    return similarity_matrix


# Split embeddings in half (TODO generalize to different numbers and sizes of networks)
def get_embeddings(combined_embed,text_length=0):
    if text_length==0:
        #n_nodes=int(combined_embed.shape[0])-20
        n_nodes = int(combined_embed.shape[0]) - 10
    else:
        n_nodes = text_length  # right now assume graphs are same size
    dim = combined_embed.shape[1]
    embed1 = combined_embed[:n_nodes]
    embed2 = combined_embed[n_nodes:]
    print(n_nodes)
    return embed1, embed2


# alignments are dictionary of the form node_in_graph 1 : node_in_graph2
# rows of alignment matrix are nodes in graph 1, columns are nodes in graph2
def score(alignment_matrix, true_alignments=None):
    if true_alignments is None:  # assume it's just identity permutation
        return np.sum(np.diagonal(alignment_matrix))
    else:
        nodes_g1 = [int(node_g1) for node_g1 in true_alignments.keys()]
        nodes_g2 = [int(true_alignments[node_g1]) for node_g1 in true_alignments.keys()]
        return np.sum(alignment_matrix[nodes_g1, nodes_g2])


def score_embeddings_matrices(embed1, embed2, topk=None, similarity_threshold=None, normalize=False,
                              true_alignments=None, sim="cosine"):
    similarity_matrix = get_embedding_similarities(embed1, embed2, sim_measure=sim)
    alignment_matrix = get_estimated_alignment_matrix(similarity_matrix, similarity_threshold, normalize)
    score = score_alignment_matrix(alignment_matrix, topk=topk, true_alignments=true_alignments)
    return score


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=50):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()


def score_alignment_matrix(alignment_matrix, topk=None, topk_score_weighted=False, true_alignments=None):
    n_nodes = alignment_matrix.shape[0]
    correct_nodes = []

    if topk is None:
        row_sums = alignment_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 10e-6  # shouldn't affect much since dividing 0 by anything is 0
        alignment_matrix = alignment_matrix / row_sums[:, np.newaxis]  # normalize

        alignment_score = score(alignment_matrix, true_alignments=true_alignments)
    else:
        alignment_score = 0
        if not sp.issparse(alignment_matrix):
            sorted_indices = np.argsort(alignment_matrix)

        for node_index in range(n_nodes):
            target_alignment = node_index  # default: assume identity mapping, and the node should be aligned to itself
            if true_alignments is not None:  # if we have true alignments (which we require), use those for each node
                target_alignment = int(true_alignments[node_index])
            if sp.issparse(alignment_matrix):
                row, possible_alignments, possible_values = sp.find(alignment_matrix[node_index])
                node_sorted_indices = possible_alignments[possible_values.argsort()]
            else:
                node_sorted_indices = sorted_indices[node_index]
            if target_alignment in node_sorted_indices[-topk:]:
                if topk_score_weighted:
                    alignment_score += 1.0 / (n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                    alignment_score += 1
                correct_nodes.append(node_index)
        alignment_score /= float(n_nodes)

    return alignment_score, set(correct_nodes)
