| Dataset              	| Weighted 	| Directed 	|  Nodes  	|   Edges   	| Diameter 	| Train time steps 	|
|----------------------	|:--------:	|:--------:	|:-------:	|:---------:	|:--------:	|:----------------:	|
| PPI                  	|     -    	|     -    	|  16,458 	|  144,033  	|    10    	|        37        	|

## PPI.
The protein-protein interactions (PPI) graph includes protein as nodes, with edges connecting two proteins for which a biological interaction was observed. Numerous experiments have been conducted to discover such interactions. These are listed in [HINTdb](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5036632/), with the list of all articles mentioning each interaction. We consider the interaction discovery date as the edge's timestamp. In a pre-processing step, we set it as the earliest publication date of its associated articles. We work in a yearly granularity between 1970 and 2015.

We publicly release this new temporal graph.
