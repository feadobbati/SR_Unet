# SR_Unet
U-net performing super-resolution of physical and biogeochemical 3D marine domains.
Both the input and the ground truth on which the U-net is trained must be numerical
model data, i.e. they are expected to be full in all their domains, with masked values
on land points.
Our use case is to improve the resolution of the Northern Adriatic Sea.
The input data are from the Copernicus Marine Service reanalysis and can be downloaded from their
[website](https://data.marine.copernicus.eu/products?facets=areas%7EMediterranean+Sea)
The target data are from the CADEAU reanalysis on the northern Adriatic Sea and can be downloaded
[here](https://zenodo.org/records/14046956).

The repository includes scripts for preprocessing, training, and evaluating the results.
Each of these phases is described below.

## Requirements
Python >= 3.10

### For the preprocessing
Scipy, numpy, netCDF4, json


### For the neural network
* Pytorch
* Pytorch lightning

### For the analysis of the results
A repository publicly available on git-hub developed inside OGS
* [bit.sea](https://github.com/inogs/bit.sea)
* Matplotlib, cmocean

## Preprocessing
The preprocessing files (stored in the preproc direcory) are:
 * interpolate_cms.py - it constructs input files by interpolating the coarse data in the high-resolution grid
 * compute_avgstd.py - it computes the average and standard deviation for each variable and save the values into a text file
 * split_train_test.py -it split the dataset into training and testset, checking that the representativity for each season is preserved
 * (join_test_train.py - if needed, the operation of splitting can be reversed)
 Since the variable names follow different conventions in CADEAU data and Copernicus Marine data, we expect to have a
 json file with the equivalences between the names in which we are interested.

## Training
As first step, we create a torch dataset with make_pt_ds.py. This create a dataset file for training data and a dataset
file for test data, associating each input to its target.
The file train.py takes as input the tensor file with training data and trains the U-net in the directory 'models/convolutional'.
At the end of its execution, the best weights found during the training are saved.


## Testing and analysing
The analysis directory includes the scripts used to evaluate the results. They work on the products of the avescan scripts, in the [bit.sea](https://github.com/inogs/bit.sea) github repository (postproc subdirectory).
Thanks to the avescan, we obtain mean, standard deviations, and other metrics for each subregion of the northern Adriatic sea, that we then use in our analysis.
