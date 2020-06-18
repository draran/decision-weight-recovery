# decision-weight-recovery
Demonstrate that complex-valued OLS regression can reliably recover the decision weights.

To recreate the figures, you need to install some software.  
1. Download/clone this repository and make note about the path/folder where you have saved it.  
2. Install miniconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/).  
3. Install the development environment used to run the analyses  
    - Open the terminal application (on MacOS/Linux) or anaconda prompt (on Windows).  
    - Navigate to the folder where you have saved this repository (i.e., PATH).  
    - Create the environment for data analyses.  
```sh
cd PATH
conda env create -f dwrenv.yml
```
4. Start jupyter notebook to run analyses.
```sh
jupyter notebook runAnalyses.ipynb
``` 
5. Run code cells one by one by pressing the run button.
6. The code should create folder called Export and will save all outputs there.

The code will both generate data and run complex-valued OLS.
Finally, the code will generate plots that are provided as supplementary materials.