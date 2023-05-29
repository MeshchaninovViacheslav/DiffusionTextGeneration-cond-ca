#!/bin/bash

offline_runs="$1./offline-run-20230528_222636-skxk*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
done