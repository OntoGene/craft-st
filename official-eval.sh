#!/bin/bash

evaldir=$(mktemp -d)

c2s="standoff2conll/conll2standoff.py"

# TODO: parallelise this loop
for path in "$@"; do
	tgtdir="$evaldir/$(basename $path)"
	mkdir "$tgtdir"
	for fn in "$path"/*.conll; do
		tgt="$tgtdir/$(basename $fn)"
		$c2s < "$fn" > "${tgt%.conll}.bionlp"
	done
done

docker run --rm -v $evaldir:/files-to-evaluate ucdenverccp/craft-eval:3.1.2_0 \
	sh -c '(cd /home/craft/evaluation && boot javac eval-concept-annotations)'

for path in "$@"; do
	cp $evaldir/$(basename $path)/*_results.tsv $path/
done
