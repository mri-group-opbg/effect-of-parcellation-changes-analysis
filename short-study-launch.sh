#!/bin/bash

DATA_DIR=$1
SUBJECT_TYPE=$2
SUBJECT=$3
NUM_TRIALS=$4
ATLAS=$5
ID=$6

PARAMETERS="--data-dir=$DATA_DIR"
PARAMETERS="$PARAMETERS --subject-id=$SUBJECT"
PARAMETERS="$PARAMETERS --subject-type=$SUBJECT_TYPE"
PARAMETERS="$PARAMETERS --num-tests=$NUM_TRIALS"
PARAMETERS="$PARAMETERS --atlas=$ATLAS"
PARAMETERS="$PARAMETERS --computation-id=$ID"

echo "launching with $PARAMETERS"

a=1
b=$((NUM_TRIALS-a))

test=1
if [ ! -f $DATA_DIR/recon_all/$SUBJECT_TYPE/$SUBJECT/$ID/random-parcels/${b}.rh.annot ]; then
	echo "parcellation computation"
	test=0
	./launch.sh 1.2.2-parcelization.py $PARAMETERS && test=1
fi

if [ $test -eq 0 ]
then
        exit
fi

test=1
if [ ! -f $DATA_DIR/recon_all/$SUBJECT_TYPE/$SUBJECT/$ID/volumes/${b}/filtered/rh.unknown.nii.gz ]; then
	echo "volume computation"
	test=0
	./launch.sh 2-volume-computation.py $PARAMETERS && test=1
fi

if [ $test -eq 0 ]
then
        exit
fi

test=1
if [ ! -f $DATA_DIR/recon_all/$SUBJECT_TYPE/$SUBJECT/$ID/correlation/${b}.pickle ]; then
	echo "correlation computation"
	test=0
	./launch.sh 3-correlation.py $PARAMETERS && test=1
fi

if [ $test -eq 0 ]
then
        exit
fi

# Alway perform graph computation since ptyhon will not produce already produced files
echo "measures computation"
./launch.sh 4.1-graph.py $PARAMETERS


