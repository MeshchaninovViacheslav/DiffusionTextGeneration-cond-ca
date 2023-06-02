#!/bin/bash

offline_runs="$1./offline-run-20230602_093448-b9d*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
done