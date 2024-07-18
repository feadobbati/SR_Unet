# SR_Unet
U-net performing super-resolution of physical and biogeochemical variables on the Northern Adriatic Sea.

## Preprocessing
The preprocessing files are:
 * interpolate_cms.py - it constructs input files by interpolating the coarse data in the high-resolution grid
 * compute_avgstd.py - it computes the average and standard deviation for each variable and save the values into a text file
 * split_train_test.py -it split the dataset into training and testset, checking that the representativity for each season is preserved
 * (join_test_train.py - if needed, the operation of splitting can be reversed)

## Training
As first step, we create a torch dataset with make_pt_ds.py, then, the U-net can be trained with train.py.
