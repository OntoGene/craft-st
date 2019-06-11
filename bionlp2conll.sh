#!/bin/bash

# Arg 1: One of {CHEBI,CL,GO_{BP,CC,MF},MOP,NCBITaxon,PR,SO,UBERON}
res=$1
# Arg 2: Target directory
target=$2
# Additional args: passed to standoff2conll.py.

CRAFT="CRAFT"
s2c="standoff2conll/standoff2conll.py"

tdir=$(mktemp -d)

cd $tdir
ln -s -t . ${CRAFT}/articles/txt/*.txt
for fn in ${CRAFT}/concept-annotation/$res/$res/bionlp/*; do
	base=$(basename $fn .bionlp)
	ln -s $fn $base.ann
	$s2c -s IOBES -c "${@:3}" ./ > $target/$base.conll
	rm $base.ann
done
