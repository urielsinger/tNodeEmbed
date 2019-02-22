# Datasets

| Dataset              	| Weighted 	| Directed 	|  Nodes  	|   Edges   	| Diameter 	| Train time steps 	|
|----------------------	|:--------:	|:--------:	|:-------:	|:---------:	|:--------:	|:----------------:	|
| arXiv hep-ph         	|     +    	|     -    	|  16,959 	| 2,322,259 	|     9    	|        83        	|
| Facebook friendships 	|     -    	|     -    	|  63,731 	|  817,035  	|    15    	|        26        	|
| Facebook wall posts  	|     +    	|     +    	|  46,952 	|  876,993  	|    18    	|        46        	|
| CollegeMsg           	|     +    	|     +    	|  1,899  	|   59,835  	|     8    	|        69        	|
| PPI                  	|     -    	|     -    	|  16,458 	|  144,033  	|    10    	|        37        	|
| Slashdot             	|     +    	|     +    	|  51,083 	|  140,778  	|    17    	|        12        	|
| Cora                 	|     -    	|     +    	|  12,588 	|   47,675  	|    20    	|        39        	|
| DBLP                 	|     +    	|     -    	| 416,204 	| 1,436,225 	|    23    	|         9        	|

## [arXiv hep-ph.](http://konect.uni-koblenz.de/networks/ca-cit-HepPh)
A research publication graph, where each node is an author and a temporal undirected edge represents a common publication with a timestamp of the publication date.
Time steps reflect a monthly granularity between March 1992 and December 1999.

## [Facebook friendships.](http://konect.uni-koblenz.de/networks/facebook-wosn-links)
A graph of the Facebook social network where each node is a user and temporal undirected edges represent users who are friends, with timestamps reflecting the date they became friends.
Time steps reflect a monthly granularity between September 2004 and January 2009.

## [Facebook wall posts.](http://konect.uni-koblenz.de/networks/facebook-wosn-wall)
A graph of the Facebook social network where each node is a user and a temporal directed edge represents a post from one user on another user's wall at a given timestamp. 
Time steps reflect a monthly granularity between September 2006 and January 2009.

## [CollegeMsg.](https://snap.stanford.edu/data/CollegeMsg.html)
An online social network at the University of California, with users as nodes and a temporal directed edge representing a private message sent from one user to another at a given timestamp. %Multiple edges with different timestamps may exist between two nodes. At a timestamp $t'$, if multiple edges exist between two users, they are grouped into a single edge weighted according to the number of original edges.
Time steps reflect a daily granularity between April 15th, 2004 and October 26th, 2004.

## [PPI.]()
The protein-protein interactions (PPI) graph includes protein as nodes, with edges connecting two proteins for which a biological interaction was observed. Numerous experiments have been conducted to discover such interactions. These are listed in HINTdb~\cite{patil2005hint}, with the list of all articles mentioning each interaction. We consider the interaction discovery date as the edge's timestamp. In a pre-processing step, we set it as the earliest publication date of its associated articles. We work in a yearly granularity between $1970$ and $2015$. 
We publicly release this new temporal graph.\textsuperscript{\ref{ppi_footnote}}

## [Slashdot.](http://konect.uni-koblenz.de/networks/slashdot-threads)
A graph underlying the Slashdot social news website, with users as nodes and edges representing replies from one user to another at a given timestamp. Time steps reflect a monthly granularity between January 2004 and September 2006.

## [Cora.](https://people.cs.umass.edu/~mccallum/data.html)
A research publication graph, where each node represents a publication, labeled with one of $L{=}10$ topical categories: artificial intelligence, data structures algorithms and theory, databases, encryption and compression, hardware and architecture, human computer interaction, information retrieval, networking, operating systems, and programming. Temporal directed edges represent citations from one paper to another, with timestamps of the citing paper's publication date. Time steps reflect a yearly granularity between 1900 and 1999.

## [DBLP.](http://dblp.uni-trier.de/xml)
A co-authorship graph, focused on the Computer Science domain. Each node represents an author and is labeled using conference keywords representing $L{=}15$ research fields: verification testing, computer graphics, computer vision, networking, data mining, operating systems, computer-human interaction, software engineering, machine learning, bioinformatics, computing theory, security, information retrieval, computational linguistics, and unknown.
Temporal undirected edges represent co-authorship of a paper, with timestamps of the paper's publication date. 
Time steps reflect a yearly granularity between 1990 and 1998.

Notice that for the arXiv hep-ph, Facebook Wall Posts, CollegeMsg, Slashdot, and DBLP graphs, multiple edges may occur from one node to another at different timestamps. Given a timestamp, if multiple edges exist, they are collapsed into a single edge, weighted according to the number of original edges, thus rendering a weighted graph, as marked in Table~\ref{table:datasets}.
