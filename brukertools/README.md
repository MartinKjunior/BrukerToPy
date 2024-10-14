# Bruker Tools

Functions used to process data from a pre-clinical Bruker MRI scanner using
ParaVision 6.0.1. The main idea is to turn the raw scanner data into python 
format (nifti files for data and ordered dictionaries for acqp, method and 
visu_pars files). Firstly run load_bruker.py through command prompt/terminal 
(instructions inside), then use bruker_to_py.py to (easily) access the data.

The data is loaded using the CLI script `load_bruker.py`:
```py
>>> cd ...\brukertools\LoadBruker
>>> python load_bruker.py "...\MRI_data" -c all
```

Once loaded, use `bruker_to_py.py`:
```py
import bruker_to_py as btp
from pathlib import Path

cwd = Path.cwd()
D_obj = btp.init(str(cwd))
```

Main issues that may be encountered using these scripts would probably have to do with paths. Let me know if you encounter issues by making a new issue and I can fix them.

### Allows to do DTI processing of bruker data using DiPy package:

**Prerequisites:**

Diffusion processing requires knowing the diffusion gradient strengths used to generate your image. These are found in the methods files that come with the diffusion data. bval is `'PVM_DwEffBval'` and bvec is `'PVM_DwGradVec'`. These need to be stored as text files, each value separated by comma. The `DataObject` from `bruker_to_py` has function `save_bval_bvec` to automatically read in, process and save bvals and bvecs. To run DiPyDTI, you need at least: your diffusion data as nifti, bvals and bvecs files.

**Example using paths to files:**
```py
# DiPyDTI can be instantiated using paths to the necessary files
# Suppose you have found the paths to your files and stored them as strings
dti = DiPyDTI()
dti.load_data(data_path, bvals_path, bvecs_path, savedir=savedir_path)
```

**Example using the bruker_to_py.py objects:**
```py
from dipy_dti_fit import DiPyDTI, DiPyPathHandler, init
path = str(Path.cwd().parent / 'MRI_data') # running from Scripts folder next to MRI_data
D_obj = init(path, msg=False)
for exam_id in D_obj.avail_exam_ids:
    path_handler = DiPyPathHandler(D_obj, exam_id)
    dpdti = DiPyDTI(path_handler.get_data_paths('dti', return_metadata=True))
    dpdti.run_pipeline(
        pipeline = ["motion_correction", "degibbs", "denoise", "fit_dti"],
        kwargs = {'degibbs':{'num_processes':3}}
        )
```

## References

`load_bruker.py` uses BrkRaw:

Lee, Sung-Ho, Ban, Woomi, & Shih, Yen-Yu Ian. (2020, June 4). BrkRaw/bruker: BrkRaw v0.3.3 (Version 0.3.3). Zenodo. http://doi.org/10.5281/zenodo.3877179

`dipy_dti_fit.py` uses DiPy:

Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I and Dipy Contributors (2014). DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics, vol.8, no.8. 
https://doi.org/10.3389/fninf.2014.00008