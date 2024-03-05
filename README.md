# Sea Surface Height reconstructions using physics based variational approaches

There are 9 folders here. 

1. Datasets: It contains original and processed data sets. Original datasets with `.nc` extensions have been processed to 
`.npy` format to be able to use them more efficiently in the context of image processing.
2. Degrading: It contains the script for the degradation of the dataset. 
3. EvaliationTable: It contains script to print the Root Mean Square Error of results. 
4. Methods: It contains the main methods that have been used in experimentation.  
    1. Conda_Vu_TV_Reg: Script of Conda Vu algorithm with Total Variation Regularization
    2. Conda_Vu_TV_Vorticity: Script of Conda Vu algorithm with composition of Total Variation and Potential Vorticity equation as a regularization term
    3. CP_TV: Script of Chambello Pock algorithm with Total Variation Regularization.
    4. CP_TV_Vorticity: Script of Chambello Pock algorithm with a combination of two regularization terms. L2 norm of Total variation and L1 norm of Potential Vorticity equations. 
5. This folder is dedicated to the different plotting scripts. 
6. PotentialVorticity: Script of calculating and plotting Potential Vorticity field from sea surface height field. 
7. results: Result folder of the experiments. 
8. Some Resulting Figures
9. utils: 
    1. `best_results.py`: This script is dedicated to choosing the top five scored results in the results folder path.
    2. `eval.py`: Evaluation functions.
    3. `fix_params_TV_Vorticity.py`: Parameters of Potential vorticity equation with total variation.
    4. `npy2mat.py`: Turns `.npy` images to `.mat`.
    5. `plot.py`: Some plottings.
    6. `preprocess.py`: Main functions of processing `.nc` to `.npy`. It takes files from `datasets/Original/dc_obs`, and saves them to `datasets/data_obs` after processing. 
    7. `toolbox.py`: It contains the main functions of the methods, from discrete gradient to criterion. 



## Notes: 
1. Methods are using the preprocessed image like objects (nxn numpy arrays)
2. The script of any method creates a folder path with the names of their methods, and hyperparameters. If it existed, it wouldn't do the experiment again. 
3. Result folders are mostly empty, expect some of them to show how they are saved and displayed. 

