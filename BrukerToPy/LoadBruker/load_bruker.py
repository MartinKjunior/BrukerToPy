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
    
    python load_bruker.py -i {path to Bruker data, enclose in "" if it contains 
                             whitespace}

The required folder with Bruker data has the naming convention 
{scan date}_{scan time}_{study id}_1_1.

The script may require admin priviledges to modify directories and files.

To load a .npy file:
    np.load(path, allow_pickle = True).item()
    '-> returns the original python object
"""

import brkraw
from argparse import ArgumentParser
import os
import numpy as np

def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error(f"The directory {arg} does not exist! If you have "
                     "whitespace in your path, enclose the path in quotes.")
    else:
        return arg
    
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
#accept command line argument for dirpath
#the required folder tends to be named "{scan date}_{scan time}_{study id}_1_1"
#                        provide the full path to this folder ^
if __name__ == "__main__":
    parser = ArgumentParser(description="Bruker loader.")
    parser.add_argument("-i", dest="dirpath", required=True,
                        help="Folder containing Bruker data.", metavar="DIR",
                        type=lambda x: is_valid_directory(parser, x))
    #vars(.parse_args) gives a dictionary of arguments, pull out dirpath
    drctry = vars(parser.parse_args())["dirpath"]
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
        newdir = make_dirs(os.path.join(f"{drctry}_loaded", "{scan_id}"))
        newdirs.append(newdir)
        acqp = pvdset.get_acqp(scan_id)
        method = pvdset.get_method(scan_id)
        np.save(os.path.join(newdir,"acqp.npy"), acqp.parameters)
        np.save(os.path.join(newdir,"method.npy"), method.parameters)
        # there is one method and acqp for the whole study,
        #  but visu_pars is unique for each reco
        for reco_id in reco_ids:
            newrecodir = os.path.join(newdir,"pdata",reco_id)
            pvdset.save_as(scan_id, reco_id, f'niiobj_{reco_id}', 
                           dir=make_dirs(newrecodir), ext='nii.gz', slope = True)
            visu_pars = pvdset.get_visu_pars(scan_id, reco_id)
            np.save(os.path.join(newrecodir,"visu_pars.npy"), visu_pars.parameters)