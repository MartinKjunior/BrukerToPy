Functions used to process data from a pre-clinical Bruker MRI scanner using
ParaVision 6.0.1. The main idea is to turn the raw scanner data into python 
format (nifti files for data and ordered dictionaries for acqp, method and 
visu_pars files). Firstly run load_bruker.py through command prompt/terminal 
(instructions inside), then use bruker_to_py.py to (easily) access the data.

load_bruker.py uses BrkRaw:
Lee, Sung-Ho, Ban, Woomi, & Shih, Yen-Yu Ian. (2020, June 4). BrkRaw/bruker: BrkRaw v0.3.3 (Version 0.3.3). Zenodo. http://doi.org/10.5281/zenodo.3877179