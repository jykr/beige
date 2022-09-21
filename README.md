# <img src="graphics/beige2.svg" alt="beige" width="150"/>
**B**ayesian variant **E**ffect **I**nference with **G**uide counts and **E**diting outcome.  

This is a generative-model-based CRISPR screen analysis software for that can account for   
*  :bar_chart: Multiple FACS sorting bins
*  :waning_gibbous_moon: Incomplete editing rate
*  :mag: Multiple target variant/bystander edit (under development)  

BEIGE models the cellular phenotype of CRISPR sorting screen data as mixture distribution. The cells will be sorted based on the theoretical quantile based on (unperturbed) control distribution and your FACS sorting quantiles. The sorted samples (red box in below schematic) are sequenced to produce the final guide counts.

<img src="graphics/model_design.svg" alt="model_design" width="500"/>

Its inference uses SVI (Stochastic Variational Inference) through [Pyro](http://pyro.ai/) to fit the posterior phenotype distribution of target element perturbation. 

## Installation 
```
pip install beret-beige
```

## Usage
See [**tutorial**](beige-tutorial.ipynb) for more information.
### CRISPR screen data without reporter information
```
beige myScreen.h5ad --prefix=my_analysis [--fit-pi|--perfect-edit|--guide_activity_column]
```
`myScreen.h5ad` must be formatted in `Screen` object in [perturb-tools](https://github.com/pinellolab/perturb-tools) package.
If you don't have reporter information measured, you can take one of three options for analysis:
1. `--fit-pi` : Editing rate is fitted so that overall likelihood of the model is maximized.
2. `--perfect-edit` : Assuming editing rate is 1 for all guides. This option is recommended over 1) based on the inference accuracy in simulation data.
3. `--guide_activity_column=your_col_name` : If you want to use external information about guide activity estimated using other software, input the guide activity in the Screen.guides DataFrame (see the [**tutorial**](beige-tutorial.ipynb))

### CRISPR screen data with reporter information
```
beige myReporterScreen.h5ad --prefix=my_analysis [--rep-pi]
```
`myReporterScreen.h5ad` must be formatted in `ReporterScreen` object in [beret](https://github.com/pinellolab/beret) package.  
*  `--rep-pi` : If you suspect your biological replicate will have overall different level of editing rates, you can let the model to fit the replicate specific scaling factor of editing rate using this option.

## Caveat
*  BEIGE assumes the phenotype distribution pre-sort sample is the same as the negative control. Whereas this assumption can be safely considered as true in case of variant screens, this may not hold true if you expect large phenotypical shift for the majority of perturbed elements.
