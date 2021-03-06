#!/bin/bash

# Arg 1: One of {CHEBI,CL,GO_{BP,CC,MF},MOP,NCBITaxon,PR,SO,UBERON}{,_EXT}
res=$1
# Arg 2: Target directory (absolute path!)
target=$2
# Additional args: passed to standoff2conll.py.

CRAFT="$(pwd)/CRAFT"
s2c="$(pwd)/standoff2conll/standoff2conll.py"

tdir=$(mktemp -d)
res_base=${res%_EXT}
res_sub=${res/_EXT/+extensions}

cd $tdir
ln -s -t . ${CRAFT}/articles/txt/*.txt
for fn in ${CRAFT}/concept-annotation/$res_base/$res_sub/bionlp/*; do
	base=$(basename $fn .bionlp)
	ln -s $fn $base.ann
	$s2c -s IOBES -c -d first-span "${@:3}" ./ > $target/$base.conll
	rm $base.ann
done
