#!/bin/bash

offline_runs="$1./offline-run-20230518_075316-0u*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
    sleep 1m
done