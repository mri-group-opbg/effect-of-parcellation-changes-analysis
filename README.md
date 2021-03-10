# Dockerized

All the software contained in this directory can be run in a dockerized RMI oriented python3 distribution:
`mastrogiovanni/rmi-python3`.

Any script has its own `launch` bash script

> ./launch-correlation.sh

```
--data-dir=/data/MASTROGIOVANNI/dati-funzionali-travelling-brain-2020 --subject-type=controllo/$place --subject-id=$item --num-tests=0
```

## Common Syntax

Any script must be launched with the following syntax:
```bash
./launch.sh <Python Script> <Argument-1> ... <Argument-n>
```

On of the arguments must always be the `--computation-id=<TOKEN>`. 
This parameter help to make unique a series of computations (labels, volumes, parcels...)

## Phase 1: Creation of Random Parcels

Create a set of randomization of default parcels (from fsaverage) according one of the two rules.
`num_test` is the number of randomization to create, `atlas` specify the Atlas to use.

```bash
./launch.sh 1.1-parcelization.py \
    --data-dir=/data/MASTROGIOVANNI/testing/ \ 
    --subject-type=controllo \
    --num-tests=0 \
    --atlas=DKTatlas40 \
    --computation-id=<TOKEN>
```

The second parcelization is performed by the script `1.2-parcelization.py`

## Phase 2: Volume Computation 

This procedure create all volumes based on parceliazation
 
```bash
./launch.sh 2-volume-computation.py \
    --data-dir=/data/MASTROGIOVANNI/testing/ \
    --subject-type=controllo \
    --num-tests=0 \
    --atlas=DKTatlas40 \
    --computation-id=<TOKEN>
```

## Phase 3: Correlation

This phase:
* filter functional data (mcflirt, compcorr, regression, lowpass)
* compute the functional masking it for each parcel 
* finally save correlation matrix of the functional signal between parcels for each randomization

```bash
./launch.sh 3-correlation.py \
    --data-dir=/data/MASTROGIOVANNI/testing/ \
    --subject-type=controllo \
    --num-tests=0 \
    --atlas=DKTatlas40 \
    --computation-id=<TOKEN>
```

## Phase 4: Algorithms

Based on different thresholds and the correlation matrix for each randomic parcelization,
this phase compute the result of a series of algorithms (83) to establish numerical and 
geometrical properties of the correlation matrix.

```bash
./launch.sh 4-graph.py \
    --data-dir=/data/MASTROGIOVANNI/testing/ \
    --subject-type=controllo \
    --num-tests=0 \
    --atlas=DKTatlas40 \
    --computation-id=par-1-DKTatlas40
```

