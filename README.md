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
```
beige your_screen.h5ad --prefix=my_analysis
```
