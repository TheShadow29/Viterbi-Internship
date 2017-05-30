#!/bin/bash
for i in {1..10}
do
	python places365_main.py $i &
	# echo $i
done
