# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:18:33 2022

@author: Martin Kozár (University of Manchester)
@contact: martin.kozar@manchester.ac.uk
"""

"""
Run in a virtual environment with the following installed:
    brkraw:    0.3.7
        python   3.7.13
        numpy    1.21.5
        shleeh   0.0.9
        nibabel  3.2.2
        pandas   1.3.4
        pillow   9.0.1
        tqdm     4.63.0
        openpyxl 3.0.9
        xlrd     2.0.1
    tkinter    8.6
    packaging  21.3
    matplotlib 3.5.1
"""

"""
How to use:
In your command line activate the appropriate virtual environment (if you use 
conda: conda activate <env name>), then
    
    cd {path to directory containing load_bruker.py}
    
    python load_bruker.py {path to Bruker data, enclose in "" if it contains 
                             whitespace} -c {one or all}

The required folder with Bruker data has the naming convention 
{scan date}_{scan time}_{study id}_1_1.

The script may require admin priviledges to modify directories and files.

To load a .npy file:
    np.load(path, allow_pickle = True).item()
    '-> returns the original python object
"""

import brkraw
from argparse import ArgumentParser, ArgumentTypeError
import os
import numpy as np
from glob import glob

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError(f"readable_dir:{path} is not a valid path")
    
def make_dirs(dirpath):
    """
    Checks if a directory exists and if not, creates it including 
    subdirectories. Useful when trying to save a file to a new directory.
    
    Parameters
    ----------
    filepath : str
        File path to save the file.
    Returns
    -------
    filepath : str
    """
    
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, mode = 0o766)
    return dirpath

def extract_paths(directories, overwrite = False):
    output = []
    for sub_dir in directories:
        if sub_dir.endswith("_loaded") and not overwrite:
            #remove the same path without _loaded
            output.remove(sub_dir[:-7])
        else:
            output.append(sub_dir)
    return output

def load_brk(drctry):
    print(f"Loading data from {drctry}")
    #brkraw takes care of the complexity of loading and verifying exams
    pvdset = brkraw.load(drctry)
    #dictionary of available scans of the form
    # {scan_id:[reco_id, reco_id...],...}
    avail = pvdset._avail
    newdirs = []
    #folder structure: {drctry}/{scan_id}/pdata/{reco_id}
    # bunch of stuff inside scan_id and reco_id, only pulling out a few things
    for scan_id, reco_ids in avail.items():
        #create a new path to save the loaded stuff to
        newdir = make_dirs(os.path.join(f"{drctry}_loaded", str(scan_id)))
        newdirs.append(newdir)
        acqp = pvdset.get_acqp(scan_id)
        method = pvdset.get_method(scan_id)
        np.save(os.path.join(newdir,"acqp.npy"), acqp.parameters)
        np.save(os.path.join(newdir,"method.npy"), method.parameters)
        # there is one method and acqp for the whole study,
        #  but visu_pars is unique for each reco
        for reco_id in reco_ids:
            newrecodir = os.path.join(newdir,"pdata",str(reco_id))
            pvdset.save_as(scan_id, reco_id, f'niiobj_{reco_id}', 
                           dir=make_dirs(newrecodir), ext='nii.gz', slope = True)
            visu_pars = pvdset.get_visu_pars(scan_id, reco_id)
            np.save(os.path.join(newrecodir,"visu_pars.npy"), visu_pars.parameters)

#accept command line argument for dirpath
#the required folder tends to be named "{scan date}_{scan time}_{study id}_1_1"
#                        provide the full path to this folder ^
if __name__ == "__main__":
    parser = ArgumentParser(
        prog="load_bruker.py",
        description="Loads Bruker data and saves it in a more accessible format."
        )
    parser.add_argument(
        "dirpath", 
        type=dir_path, 
        help="Path to the directory containing the Bruker data. If count is 'all'," \
            + " this should be the parent directory of the folders containing the Bruker data.", 
        metavar="dir"
        )
    parser.add_argument(
        "-c",
        "--count",
        type=str,
        default="one",
        help="How to load the Bruker data. Options: 'one' (folder) or 'all' (folders). Default: 'one'.",
        required=False,
        choices=["one", "all"]
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing '_loaded' folders. Default: False.",
        required=False
    )
    args = parser.parse_args()
    drctry = args.dirpath
    count = args.count
    overwrite = args.overwrite
    
    if count == "one":
        load_brk(drctry)
    elif count == "all":
        directories = extract_paths(glob(drctry + "/*"), overwrite)
        for sub_drctry in directories:
            load_brk(sub_drctry)