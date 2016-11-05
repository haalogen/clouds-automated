#!/bin/bash

for i in `seq 1 30`;
do
	echo "====== Experiment" $i "======"
	python Find-Cloud-2Images-Diff-Brute.py
done 
