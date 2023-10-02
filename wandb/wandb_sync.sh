#!/bin/bash

offline_runs="./$1*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
done