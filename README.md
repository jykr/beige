# <img src="beige2.svg" alt="beige" width="150"/>
**B**ayesian variant **E**ffect **I**nference with **G**uide counts and **E**diting outcome.  

This is a CRISPR screen analysis software for that can account for 
*  Multiple FACS sorting bins
*  Incomplete editing rate
*  Multiple target variant/bystander edit (under development)  


## Installation 
```
git clone https://github.com/jykr/beige
cd beige/
pip install -e .
```
This requires pre-existing numpy installation.

## Usage
### CRISPR screen data without reporter information
```
beige myScreen.h5ad --prefix=my_analysis [--fit-pi|--perfect-edit|--guide_activity_column]
```
`myScreen.h5ad` must be formatted in `Screen` object in [perturb-tools](https://github.com/pinellolab/perturb-tools) package.
If you don't have reporter information measured, you can take one of three options for analysis:
1. `--fit-pi` : Editing rate is fitted so that overall likelihood of the model is maximized.
2. `--perfect-edit` : Assuming editing rate is 1 for all guides. This option is recommended over 1) based on the inference accuracy in simulation data.
3. `--guide_activity_column=your_col_name` : If you want to use external information about guide activity estimated using other software, input the guide activity in the Screen.guides DataFrame (see `Screen` object in [perturb-tools](https://github.com/pinellolab/perturb-tools)).Pass the column name as the argument.(under development)

### CRISPR screen data with reporter information
```
beige myReporterScreen.h5ad --prefix=my_analysis [--rep-pi]
```
`myReporterScreen.h5ad` must be formatted in `ReporterScreen` object in [beret](https://github.com/pinellolab/beret) package.  
*  `--rep-pi` : If you suspect your biological replicate will have overall different level of editing rates, you can let the model to fit the replicate specific scaling factor of editing rate using this option.


