
1. ./gen_graph.py to generate the images
2. ./gen_graph.py to generate ./graph.pbtxt, which is the graph proto
3. ../models/hex/scripts/augment_exclusion.py to augment_exclusion edges, generating ./graph.pbtxt.excl
4. ../models/hex/scripts/utils.py to generate junction trees, generating ./graph.pbtxt.excl.pb*
5. link ./graph.pbtxt to data/graph.pbtxt
6. link ./graph.pbtxt.excl.pb to data/jtree.pb
