# Bruker Tools

Functions used to process data from a pre-clinical Bruker MRI scanner using
ParaVision 6.0.1. The main idea is to turn the raw scanner data into python 
format (nifti files for data and ordered dictionaries for acqp, method and 
visu_pars files). Firstly run load_bruker.py through command prompt/terminal 
(instructions inside), then use bruker_to_py.py to (easily) access the data. 

***Warning:*** `load_bruker.py` was only tested with python=3.7 and bruker=0.3.7. `dipy_dti_fit.py` requires at least python=3.10.

The data is loaded using the CLI script `load_bruker.py`:
```py
>>> cd ...\brukertools\LoadBruker
>>> python load_bruker.py "...\MRI_data" -c all
```

Once loaded, use `bruker_to_py.py`:
```py
import bruker_to_py as btp
from pathlib import Path

cwd = Path.cwd().parent / 'MRI_data' # Should be the path to Bruker data, not the script
D_obj = btp.init(
    str(cwd), # path to the data folder (contains Bruker and _loaded folders)
    msg=False, # whether to show the data structure (exam_id, scan_id, reco_id)
    animal_overview="animal_overview.xlsx" # if following the folder structure below, filename is enough, otherwise provide the absolute path
)
```
animal_overview should be a csv or excel file with at least 2 columns with different animal's exam IDs and the scan IDs of the diffusion scans for DTI.

## ID definitions

- **Exam ID**: The MR number, number of a new exam card in paravision, e.g. 230215.
- **Scan ID**: The number of the scan, e.g. 10, shown in paravision as E10 on the exam card.
- **Reco ID**: The reconstruction number, e.g. 1. The default image is 1, any additional processing, such as returning phase images, will have a different reco id. Diffusion images are usually reco id 1.

## DTI processing of bruker data using DiPy package:

### Prerequisites:

Diffusion processing requires knowing the diffusion gradient strengths used to generate your image. These are found in the methods files that come with the diffusion data. bval is `'PVM_DwEffBval'` and bvec is `'PVM_DwGradVec'`. These need to be stored as text files, each value separated by comma. The `DataObject` from `bruker_to_py` has function `save_bval_bvec` to automatically read in, process and save bvals and bvecs. To run DiPyDTI, you need at least: your diffusion data as nifti, bvals and bvecs files (a brain mask is optional to save time computing the DTI fit).

```py
D_obj.save_bval_bvec(
    identifier='dti', # name of the column with scan IDs of diffusion data
    exam_col='MR #', # name of the column with exam IDs of the animals
    round=True # whether to round the bvals to the nearest 100
)
```

`DiPyPathHandler` from `path_handler.py` provides a convenient way of fetching the required data paths. It requires a csv-like file with at least 2 columns 
showing the animal's exam_id and scan_id to identify the DTI datasets. This is 
the third input for `btp.init()`, or second input to `btp.DataObject()`.

### Usual folder structure:

```
Study_PRIME000/
├── animal_overview.xlsx
├── Processed_MRI_data/
│   └── 20230426_102559_230178_1_1/
│       ├── DiPyDTI/
│       └── BrainMask/
├── MRI_data/
│   ├── 20230426_102559_230178_1_1/
│   └── 20230426_102559_230178_1_1_loaded/  <- from load_bruker.py
└── Scripts/
    ├── bruker_to_py.py
    ├── dipy_dti_fit.py
    ├── path_handler.py
    ├── multiprocess_wrappers.py
    └── main.ipynb
```

Additional subfolders can be added to each exam in `Processed_MRI_data` using 
```py
D_obj = bruker_to_py.init('<path to /MRI_data>')
D_obj.prepare_savedirs('BrainMask')
```

**1. Example using paths to files:**
```py
# DiPyDTI can be instantiated using paths to the necessary files
# Suppose you have found the paths to your files and stored them as strings
# A new subfolder "DiPyDTI" will be created in savedir to store the outputs
from dipy_dti_fit import DiPyDTI
dpdti = DiPyDTI()
dpdti.load_data(
    data_path, bvals_path, bvecs_path, savedir=savedir_path, exam_id=230215, 
    scan_id=6
)
dpdti.run_pipeline()
```

**2. Example using the bruker_to_py.py objects (recommended):**

* requires `bruker_to_py`

```py
from dipy_dti_fit import DiPyDTI
from path_handler import DiPyPathHandler, init
path = str(Path.cwd().parent / 'MRI_data') # running from Scripts folder next to MRI_data
D_obj = init(path, msg=False, animal_overview="animal_overview.xlsx")
for exam_id in D_obj.avail_exam_ids:
    path_handler = DiPyPathHandler(D_obj, exam_id)
    dpdti = DiPyDTI(
        path_handler.get_data_paths(
            dti_col='dti', # name of the column with scan IDs of diffusion data
            id_col='MR #', # name of the column with exam IDs of the animals
            return_metadata=True
        )
    )
    dpdti.run_pipeline(
        pipeline = ["motion_correction", "degibbs", "denoise", "fit_dti"],
        kwargs = {'degibbs':{'num_processes':3}}
    )
```

**3. Example of running DiPyDTI on multiple cores in parallel:**

* requires `mpire`

```py
from multiprocess_wrappers import multiprocess_DTI
path = str(Path.cwd().parent / 'MRI_data')
multiprocess_DTI(path, dti_col = 'dti', id_col = 'MR #', n_jobs = 3)
```

## Recipes

### Loading a pickled object (methods, acqp, visu_pars, TensorFit... ending .npy) 

```py
import numpy as np
path_to_object = R"..." # R means "raw string", so \ is not treated as a special symbol
python_object = np.load(path_to_object, allow_pickle=True).item()
```

### Loading a nifti image

```py
import nibabel as nib
path_to_nii = R"..." # R means "raw string", so \ is not treated as a special symbol
nifti = nib.load(path_to_nii)
```

### ***bruker_to_py***

### Loading an unprocessed file

 * if you only need the path without loading the data, use `only_path=True` (overrides `as_numpy=`)

```py
data_dict = D_obj.pull_exam_data(
    230215, # exam ID
    6, # scan ID
    1, # reco ID
    as_numpy = False, # False returns nifti1image, otherwise ndarray
    only_path = False
)
data_dict['recos']['data'] # to access the image
```

 * returns a dictionary with `dict_keys(['exam_id', 'scan_id', 'acqp', 'method', 'recos'])` and in `['recos']` there are `dict_keys(['path', 'data', 'visu_pars'])` (`'data'` missing if `only_path=True`).

### Loading a processed file

 * if you only need the path without loading the data, use `only_path=True` (overrides `to_dict=` and `as_numpy=`)

```py
data_collection = D_obj.pull_processed_data(
    230215, # exam ID
    ['DiPyDTI', '*', 'DTIFit'], # path to the data if nested, otherwise use a single string
    to_dict = False, # if True, returns {filename: <loaded object>}
    as_numpy = False, # False returns nifti1image, otherwise ndarray
    substring = '9_mean_diffusivity', # a part of the filename that identifies the file, returns all that match
    only_path = False
)
data_collection[-1] # to access the image if to_dict=False
data_collection[filename] # to access the image if to_dict=True
```

## References

`load_bruker.py` uses [BrkRaw](https://github.com/BrkRaw/brkraw):

Lee, Sung-Ho, Ban, Woomi, & Shih, Yen-Yu Ian. (2020, June 4). BrkRaw/bruker: BrkRaw v0.3.3 (Version 0.3.3). Zenodo. http://doi.org/10.5281/zenodo.3877179

`dipy_dti_fit.py` uses [DiPy](https://github.com/dipy/dipy):

Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I and Dipy Contributors (2014). DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics, vol.8, no.8. 
https://doi.org/10.3389/fninf.2014.00008