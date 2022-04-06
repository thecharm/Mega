import numpy as np
import argparse
import networkx as nx
import time
import os
import sys

from stanfordcorenlp import StanfordCoreNLP

try: import cPickle as pickle
except ImportError:
	import pickle
from scipy.sparse import csr_matrix

import xnetmf
from config import *
from alignments import *

def parse_args(path):
	parser = argparse.ArgumentParser(description="Run REGAL.")

	parser.add_argument('--input', nargs='?', default=path, help="Edgelist of combined input graph")

	parser.add_argument('--output', nargs='?', default='emb/arenas990-1.emb',
	                    help='Embeddings path')

	parser.add_argument('--attributes', nargs='?', default='data/attributes/attr1-2vals-prob0.000000',
	                    help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

	parser.add_argument('--attrvals', type=int, default=2,
	                    help='Number of attribute values. Only used if synthetic attributes are generated')


	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--k', type=int, default=10,
	                    help='Controls of landmarks to sample. Default is 10.')

	parser.add_argument('--untillayer', type=int, default=2,
                    	help='Calculation until the layer for xNetMF.')
	parser.add_argument('--alpha', type=float, default = 0.01, help = "Discount factor for further layers")
	parser.add_argument('--gammastruc', type=float, default = 1, help = "Weight on structural similarity")
	parser.add_argument('--gammaattr', type=float, default = 1, help = "Weight on attribute similarity")
	parser.add_argument('--numtop', type=int, default=5,help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
	parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
	return parser.parse_args()

def main(args,path_save,l):
	dataset_name = args.output.split("/")
	if len(dataset_name) == 1:
		dataset_name = dataset_name[-1].split(".")[0]
	else:
		dataset_name = dataset_name[-2]

	#Get true alignments
	true_alignments_fname = args.input.split("_")[0] + "_edges-mapping-permutation.txt" #can be changed if desired
	print ("true alignments file: ", true_alignments_fname)
	true_alignments = None
	if os.path.exists(true_alignments_fname):
		with open(true_alignments_fname, "rb") as true_alignments_file:
			true_alignments = pickle.load(true_alignments_file, encoding='iso-8859-1')

	#Load in attributes if desired (assumes they are numpy array)
	if args.attributes is not None:
		args.attributes = np.load(args.attributes) #load vector of attributes in from file
		print (args.attributes.shape)

	#Learn embeddings and save to output
	print ("learning representations...")
	before_rep = time.time()
	embed=learn_representations(args,l)
	print(embed.shape)
	after_rep = time.time()
	print("Learned representations in %f seconds" % (after_rep - before_rep))

	#Score alignments learned from embeddings
	#embed = np.load(args.output,allow_pickle=True)
	emb1, emb2 = get_embeddings(embed)
	print(emb1.shape)
	print(emb2.shape)
	before_align = time.time()
	if args.numtop == 0:
		args.numtop = None
	alignment_matrix = get_embedding_similarities(emb1, emb2, num_top = args.numtop)
	print(alignment_matrix.shape)
	print(alignment_matrix)
	#alignment_matrix_complete=alignment_matrix.A
	#print(alignment_matrix_complete.shape)
	if args.numtop == 0:
		np.savetxt(path_save,alignment_matrix)
	else:
		np.savetxt(path_save, alignment_matrix.A)
	'''
	#Report scoring and timing
	after_align = time.time()
	total_time = after_align - before_align
	print("Align time: ", total_time)

	if true_alignments is not None:
		topk_scores = [1,5,10,20,50]
		for k in topk_scores:
			score, correct_nodes = score_alignment_matrix(alignment_matrix, topk = k, true_alignments = true_alignments)
			print("score top%d: %f" % (k, score))

	print(dataset_name)
	'''
#Should take in a file with the input graph as edgelist (args.input)
#Should save representations to args.output
def learn_representations(args,l):
	nx_graph = nx.read_edgelist(args.input, nodetype = int, comments="%")
	print ("read in graph")

	adj = nx.adjacency_matrix(nx_graph, nodelist=l).todense()
	graph = Graph(adj, node_attributes = args.attributes)
	max_layer = args.untillayer
	if args.untillayer == 0:
		max_layer = None
	alpha = args.alpha
	num_buckets = args.buckets #BASE OF LOG FOR LOG SCALE
	if num_buckets == 1:
		num_buckets = None
	rep_method = RepMethod(max_layer = max_layer, 
							alpha = alpha, 
							k = args.k, 
							num_buckets = num_buckets, 
							normalize = True, 
							gammastruc = args.gammastruc, 
							gammaattr = args.gammaattr)
	if max_layer is None:
		max_layer = 1000
	print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
	representations = xnetmf.get_representations(graph, rep_method)
	pickle.dump(representations, open(args.output, "wb+"))
	return representations
		

if __name__ == "__main__":

	path_load = ''
	filelist = os.listdir(path_load)
	path_save = ''
	i = 0
	for item in filelist:
		l = []
		f = open(os.path.join(path_load, item), 'r', encoding='utf-8')
		data = f.readlines()
		for j in range(len(data) + 11):
			l.append(j)
		print(l)
		args = parse_args(path_load + item) # 读取联合图
		main(args, path_save + item, l)

