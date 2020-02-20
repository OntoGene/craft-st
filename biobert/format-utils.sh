#!/bin/bash


craft2bert () {
	# CHEBI, CL_EXT etc.
	etype=$1
	# spans or ids
	level=$2
	# "test" or "train dev"
	subset=$3

	python3 biocodes/conll2conll.py bert -l $level \
		-t data/data/$level/$etype \
		-c ../labeled.feat/$etype \
		-s ../splits.subm.json \
		-S $subset
		# -a ../abbrevs.json
}

mk_tagset() {
	echo O-NIL > $1/tag_set.txt
	cut -f 2 $1/train_dev.tsv | grep '^I' | sort -u >> $1/tag_set.txt
}

mk_tagset_pretrained() {
	echo O-NIL > $1/tag_set_pretrained.txt
	ln -s -t $2 $(realpath --relative-to $2 $1/tag_set_pretrained.txt)
	cut -f 2 {$1,$2}/train_dev.tsv | grep '^I' | sort -u >> $1/tag_set_pretrained.txt
}

bert2craft() {
	# CHEBI, CL_EXT etc.
	etype=$1
	# spans-first, spans-only, ids-first, ids-only
	harm=$2
	# "" or "pretrained-"
	pret=$3

	python3 biocodes/conll2conll.py craft \
		-t data/pred/${harm}${pret:+-pret}/$etype \
		-c ../labeled.feat/$etype \
		-p data/tmp/spans-$etype \
		-i data/tmp/${pret}ids-${etype}${pret:+.1000} \
		-m $harm \
		-s ../splits.subm.json \
		-S test
		# -a ../abbrevs.json
}
