#!/bin/bash

evaldir=$(mktemp -d)

workers=4
c2s="standoff2conll/conll2standoff.py"
corpus="/tmp/craft-st-2019"

for path in "$@"; do
	tgtdir="$evaldir/$(basename $path)"
	mkdir "$tgtdir"
	ls "$path"/*.conll | parallel -j $workers "cut -f 1-4 {} | sed 's/[BIES]-NIL$/O-NIL/' | $c2s > $tgtdir/{/.}.bionlp"
done

docker run --rm -v $evaldir:/files-to-evaluate -v $corpus:/corpus-distribution \
	ucdenverccp/craft-eval:3.1.3_0.1.2 \
	sh -c '(cd /home/craft/evaluation && boot javac eval-concept-annotations -c /corpus-distribution -i /files-to-evaluate -g /files-to-evaluate)'

for path in "$@"; do
	cp $evaldir/$(basename $path)/*_results.tsv $path/
done
