# Spontaneous maps
Repository created for the bachelor thesis `Extraction of spontaneously ocurring activity patterns from an electrophysiological signal` \
The purpose of this code is 
1. inference of functional maps from spontaneous activity.
2. plotting figures for the thesis

## Dependencies
Code dependencies are specified in `requirements.txt` and can be installed with by in terminal: \
\
```pip install -r requirements.txt```

## Workflow
1. **correlation_maps** - extract activation patterns and calculate (and interpolate) channel's correlation maps
2. **spontaneous_map** - create spontaneous map from computed correlation maps
3. **test_spontaneous_map** - test statistical significance of spontaneous map similarity to evoked map

## Running analysis
To run analysis of single array: \
```./run.sh $monkey $Array_ID $method```
- monkey - "L", "A"
- Array_ID - 1-16
- method - "MUA", "MUAe", "LFP", "nLFP"

To run parameter search with parameter combinations saved in `parameter_space.pkl`: \
```./scan ``` \
Monkey, arrays and others are defined in the bash script.

## Data set
All electrophysiological data is available at https://gin.g-node.org/NIN/V1_V4_1024_electrode_resting_state_data 