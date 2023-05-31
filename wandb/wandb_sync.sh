#!/bin/bash

offline_runs="$1./offline-run-20230530_171739-av7*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
done