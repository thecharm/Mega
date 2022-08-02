# Code for structual graph alignment
This module is used to generate the structual alignment map for our efficient graph alignment.

## Requirements
* `stanford-corenlp==4.2.0`
* `nltk==3.4.5`
* `scipy==1.4.1`

## Usage
Run xnetmf.py for retrieving the structural graph, Then run StructuralAlign.py for graph similarity map.

Please change the path for input texts and corresponding visual scene graphs in xnetmf.py (in the directory `data`).

We also provide the aligned graph in the directory `combined_graph_edges` for your convenience.




