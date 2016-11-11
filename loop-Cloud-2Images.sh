#!/bin/bash

echo "############## Shot Time: " $1 " ################"
for i in `seq 1 50`;
do

	echo "====== Experiment" $i "======"
	python -W ignore Find-Cloud-2Images-FFT-Quant.py $1
done 
