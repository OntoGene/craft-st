#!/bin/bash

evaldir=$(mktemp -d)

workers=4
c2s="standoff2conll/conll2standoff.py"

for path in "$@"; do
	tgtdir="$evaldir/$(basename $path)"
	mkdir "$tgtdir"
	ls "$path"/*.conll | parallel -j $workers "$c2s < {} > $tgtdir/{/.}.bionlp"
done

docker run --rm -v $evaldir:/files-to-evaluate ucdenverccp/craft-eval:3.1.3_0.1.2 \
	sh -c '(cd /home/craft/evaluation && boot javac eval-concept-annotations)'

for path in "$@"; do
	cp $evaldir/$(basename $path)/*_results.tsv $path/
done
