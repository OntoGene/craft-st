#!/bin/bash

# Collect the last line from results/*_results.tsv.

for res in {CL,CHEBI,GO_{BP,CC,MF},MOP,NCBITaxon,PR,SO,UBERON}; do
	res_lc=$(python3 -c "print('$res'.lower(), end='')")
	echo -ne "\tAA"  # move this line to top in sorting
	head -n 1 results/${res_lc}_results.tsv
	echo -ne "$res\t"
	tail -n 1 results/${res_lc}_results.tsv
done | sort -u | cut -f 1,3-
