#!/bin/bash



TEXT=$1

cat $TEXT | sed "s/ /\n/g" | sort -u | awk 'BEGIN { n = 0;} { print n,$1; n+=1}' > vocab.for.cued.txt


#sort -u tokens.txt | awk 'BEGIN { n = 0;} { print n,$1; n+=1}' > vocab.for.cued.txt

