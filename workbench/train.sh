#!/bin/bash

export LD_LIBRARY_PATH=cuda/lib64:cuda/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

TRAINSET=train.txt
DEVSET=dev.txt

INPUTVOCAB=vocab.with.unk.cued.txt
OUTPUTVOCAB=$INPUTVOCAB

INPUTLAYER=$(( $(cat $INPUTVOCAB | wc -l) + 2))
OUTPUTLAYER=$(( $(cat $OUTPUTVOCAB | wc -l) + 2))

MODELPATH=minimodel

mkdir -p $MODELPATH

MODEL=$MODELPATH/model

echo "Input vocab. size: "$INPUTLAYER
echo "Output vocab. size: " $OUTPUTLAYER

FULLVOCSIZE=$INPUTLAYER

./../cued-rnnlm.v1.1/rnnlm.cued.v1.1 -train \
	-trainfile $TRAINSET \
	-validfile $DEVSET \
	-device 0 \
	-minibatch 16 \
	-chunksize 6 \
	-layers $INPUTLAYER:64i:128m:$OUTPUTLAYER \
	-traincrit nce \
	-ncesample 100 \
	-lognormconst 9.0 \
	-lrtune newbob \
	-learnrate 0.2 \
	-inputwlist $INPUTVOCAB \
	-outputwlist $OUTPUTVOCAB \
	-randseed 1 \
	-writemodel $MODEL \
	-independent 1 \
	-min_improvement 1.003 \
	-fullvocsize $FULLVOCSIZE \
	-debug 3
