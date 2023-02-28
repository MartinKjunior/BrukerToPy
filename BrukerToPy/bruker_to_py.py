# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:02:46 2022

@author: mbcxamk2

This is a script meant for loading Bruker data into Python. The Bruker data 
should be in the format resulting from using load_bruker() which utilises
BrkRaw (pip install bruker) for loading and saving the data.
"""

import os
import pprint
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from glob import glob

def load(path, msg = True):
    '''
    Loads in data from folders containing niftis and pickled metadata.

    Parameters
    ----------
    path : str
        Path to the '..._loaded' folder holding the loaded data.
    msg : bool, optional
        Print out the available study_id and reco_id. The default is True.

    Raises
    ------
    FileNotFoundError
        Invalid path.

    Returns
    -------
    BrPyLoader
        Object for retreiving data from the '..._loaded' subfolders. 
        Similar behaviour to the pvdset object from BrkRaw.

    '''
    if os.path.isdir(path):
        return BrPyLoader(path, msg)
    raise FileNotFoundError(path)

class BrPyLoader:
    """
    Class for holding path and scan info and retreiving raw data.
    
    Attributes:
        path - string containing path to the Bruker data folder
        scan_ids - list of integers corresponding to available scans
        available_scans - dictionary of the form {scan_id:[reco_id,...]} (ints)
        
    Methods:
        get_{acqp|method|visu_pars} - gets appropriate metadata -> OrderedDict
        get_dataobj - gets either nifti or (numpy array, affine, header)
        get_all_recos - gets all recos under a study_id
        get_all_scans - gets all data under an exam_id (all studies and recos)
        get_dwi_metadata - gets information relevant to dwi processing
    """
    
    def __init__(self, path, msg = True):
        if os.path.basename(path) == 'Scripts':
            self.path = os.path.dirname(path)
        else:
            self.path = path
        self.available_scans = {}
        self._find_available_scans()
        if msg:
            print("Successfully loaded the data. The available scans are:")
            pprint.pprint(self.available_scans)
    
    def _find_available_scans(self):
        paths = glob(f"{self.path}/*")
        try:
            self.scan_ids = sorted([int(os.path.basename(path)) for path in paths])
        except Exception as e:
            print(e)
            raise ValueError("All scan_ids should be integers.") from e
    
        for scan_id in self.scan_ids:
            self.available_scans[scan_id] = []
            recos = glob(f"{self.path}/{scan_id}/pdata/*")
            for reco in recos:
                self.available_scans[scan_id].append(int(os.path.basename(reco)))
    
    def _check_id_validity(self, *args):
        if len(args) > 2:
            raise ValueError('Too many ids. Please pass one or two ids.')
        if len(args) == 0:
            raise ValueError('No ids passed. Please pass one or two ids.')
        #args[0] -> scan_id, args[1] -> reco_id
        if args[0] in self.scan_ids:
            if len(args) == 2:
                if args[1] in self.available_scans[args[0]]:
                    return True
                raise ValueError(f"Invalid reco_id. For scan_id {args[0]} " 
                                     "the available reco_ids are "
                                     f"{self.available_scans[args[0]]}")
            return True
        raise ValueError(f"Invalid scan_id {args[0]} available scan_ids are "
                             f"{self.scan_ids}")
    
    def get_acqp(self, scan_id):
        if self._check_id_validity(scan_id):
            return np.load(f"{self.path}/{scan_id}/acqp.npy", 
                           allow_pickle = True).item()
    
    def get_method(self, scan_id):
        if self._check_id_validity(scan_id):
            return np.load(f"{self.path}/{scan_id}/method.npy", 
                           allow_pickle = True).item()
    
    def get_visu_pars(self, scan_id, reco_id):
        if self._check_id_validity(scan_id, reco_id):
            return np.load(f"{self.path}/{scan_id}/pdata/{reco_id}/visu_pars.npy",
                           allow_pickle = True).item()
    
    def get_dataobj(self, scan_id, reco_id, as_numpy = True):
        '''
        Fetches all data contained within a nifti file specified by scan_id and
        reco_id.

        Parameters
        ----------
        scan_id : int
        reco_id : int
        as_numpy : bool, optional
            Return a tuple containing a numpy array. The default is True.

        Raises
        ------
        FileNotFoundError
            No nifti files found.

        Returns
        -------
        nib.nifti1.Nifti1Image or tuple(np.ndarray, nib.nifti1.Nifti1Header, np.ndarray)
            Return type depends on as_numpy.

        '''
        if self._check_id_validity(scan_id, reco_id):
            path = f"{self.path}/{scan_id}/pdata/{reco_id}"
            paths = glob(f"{path}/*")
            paths = [x for x in paths if x.endswith((".nii", ".nii.gz"))]
            if len(paths) == 1:
                if as_numpy:
                    niftiimage = nib.load(paths[0])
                    return (niftiimage.get_fdata(), 
                            niftiimage.affine,
                            niftiimage.header)
                return (nib.load(paths[0]), )
            elif len(paths) > 1:
                print(f"Multiple datasets found in {reco_id=}.")
                output = tuple()
                for _path in paths:
                    if as_numpy:
                        niftiimage = nib.load(_path)
                        output += (niftiimage.get_fdata(), 
                                   niftiimage.affine,
                                   niftiimage.header)
                    else:
                        output += (nib.load(_path), )
            else:
                raise FileNotFoundError('Failed to find path to data.')
            return (output[::2], output[1::2])
    
    def get_all_recos(self, scan_id, as_numpy = True):
        """
        Returns a dictionary containing all records under a given scan id.

        Parameters
        ----------
        scan_id : int
        as_numpy : bool, optional
            The dictionary get populated with either numpy arrays or nifti 
            images. The default is True which returns a dictionary of arrays.

        Returns
        -------
        output : dict
            A dictionary of the form {reco_id: numpy.ndarray or Nifti1Image}.

        """
        if self._check_id_validity(scan_id):
            output = {}
            for reco_id in self.available_scans[scan_id]:
                path = f"{self.path}/{scan_id}/pdata/{reco_id}"
                flag = True
                for _path in glob(f"{path}/*"):
                    if _path.endswith(".nii.gz"):
                        reco_id2 = os.path.basename(_path)[7:-7]
                        if reco_id2.isdecimal():
                            reco_id2 = int(reco_id2)
                        elif flag:
                            print(f"User warning! The folder {path} contains "
                                  "more than one image. The reco_id key type "
                                  "will be changed to str.")
                            flag = False
                        output[reco_id2] = {}
                        if as_numpy:
                            niftiimg = nib.load(_path)
                            output[reco_id2]["data"] = niftiimg.get_fdata()
                            output[reco_id2]["affine"] = niftiimg.affine
                            output[reco_id2]["header"] = niftiimg.header
                        else:
                            output[reco_id2]["data"] = nib.load(_path)
                        output[reco_id2]["visu_pars"] = self.get_visu_pars(
                            scan_id, reco_id
                            )
            return output
    
    def get_all_scans(self, as_numpy = True):
        """
        Returns a dictionary containing all records under all available scan 
        ids. Warning: may be very slow!

        Parameters
        ----------
        as_numpy : bool, optional
            The dictionary gets populated with either numpy arrays or nifti 
            images. The default is True which returns a dictionary of arrays.

        Returns
        -------
        output : dict
            A dictionary of the form {reco_id: numpy.ndarray or Nifti1Image}.

        """
        output = {}
        for scan_id, reco_ids in self.available_scans.items():
            output[scan_id] = {}
            for reco_id in reco_ids:
                path = (f"{self.path}/{scan_id}/pdata/{reco_id}/"
                        "niiobj_{reco_id}.nii.gz")
                try:
                    if as_numpy:
                        output[scan_id][reco_id] = nib.load(path).get_fdata()
                    else:
                        output[scan_id][reco_id] = nib.load(path)
                except FileNotFoundError:
                    print(f"User warning! The folder {os.path.dirname(path)} "
                          "contains more than one image. The reco_id key type "
                          "will be changed to str.")
                    for _path in glob(f"{os.path.dirname(path)}/*"):
                        if _path.endswith(".nii.gz"):
                            key = os.path.basename(_path)[:-7].split("_")[1]
                            if as_numpy:
                                output[scan_id][key] = nib.load(
                                    _path
                                    ).get_fdata()
                            else:
                                output[scan_id][key] = nib.load(_path)
        return output
    
    def get_dwi_metadata(self, scan_id, reco_id):
        """
        Pulls out a selection of metadata relevant to DWI MRI signal 
        processing.

        Parameters
        ----------
        scan_id : int
        reco_id : int

        Returns
        -------
        dict
            Dictionary containing the metadata.

        """
        if self._check_id_validity(scan_id, reco_id):
            visu_pars = self.get_visu_pars(scan_id, reco_id)
            method = self.get_method(scan_id)
            
            return \
                dict(
                    data_slope    = visu_pars['VisuCoreDataSlope'],
                    data_offset   = visu_pars['VisuCoreDataOffs'],
                    num_avrgs     = visu_pars['VisuAcqNumberOfAverages'],
                    num_b_vals    = method['PVM_DwNDiffExpEach'],
                    grad_dur      = method['PVM_DwGradDur'],
                    grad_sep      = method['PVM_DwGradSep'],
                    num_of_dirs   = method['PVM_DwNDiffDir'],
                    num_b0_imgs   = method['PVM_DwAoImages'],
                    b_nominal     = method['PVM_DwBvalEach'],
                    b_effective   = method['PVM_DwEffBval'],
                    n_repetitions = method['PVM_NRepetitions']
                )

class DataObject:
    '''
    Class for handling processed data.
    
    Attributes:
        
    
    '''
    #To do: make pull_exam_data return either a dataclass or a named tuple for
    #convenience when handling the data.
    def __init__(self, path = ""):
        if not os.path.isdir(path):
            raise ValueError(f"Invalid path input: {path}")

        self.path            = path
        self.exam            = None
        self.exam_data       = None
        self.current_exam_id = None
        self.avail_exams     = {}
        self.scan_data       = []
        self.reco_data       = []
        self.processing_id   = None
        self.processing_path = None
        
        (self.paths, 
         self.paths_loaded, 
         self.paths_raw)     = self._get_paths()
        self.avail_exam_ids  = self._get_exam_ids()
        self.savedirs        = self._prepare_savedir_paths()
        self.processed_dirs  = self._find_processed_dirs()
        self.rat_overview    = self._get_rat_overview()
        #isinstance(n, self.int) will accept any integer now
        self.int = (int, np.integer)
        
    def _get_paths(self):
        '''
        Parameters
        ----------
        path : str
            Path to folder containing all Bruker folders (Bruker folder names 
            have the form "{scan date}_{scan time}_{exam id}_1_1".

        Returns
        -------
        all_p : list
            All Bruker folders within path.
        p_loaded : list
            Bruker folders loaded into niftis.
        p_raw : list
            Raw Bruker folders.

        '''
        all_p = glob(f"{self.path}/*")
        p_loaded = [x for x in all_p if x.endswith("_loaded")]
        p_raw = [x for x in all_p if x not in p_loaded]
        
        return (all_p, p_loaded, p_raw)
    
    def _get_exam_ids(self):
        '''
        Returns
        -------
        list[str]
            List of {exam id}s from the Bruker folder names.

        '''
        return [os.path.basename(x).split('_')[-3] for x in self.paths_raw]
    
    def _prepare_savedir_paths(self):
        '''
        Create folders for saving data and store their paths.

        Returns
        -------
        None.

        '''
        savedirs = []
        for path in self.paths_raw:
            p = Path(path)
            parts = list(p.parts)
            parts[-2] = 'Processed_' + parts[-2]
            savedir = os.path.join(*parts)
            if not os.path.isdir(savedir):
                os.makedirs(savedir, mode = 0o766)
            savedirs.append(savedir)
        return savedirs
    
    def _find_processed_dirs(self):
        '''
        Finds the folder names where I'm saving processed data.

        Returns
        -------
        None.

        '''
        dirs = glob(f'{self.savedirs[0]}/*')
        basenames = [os.path.basename(p) for p in dirs]
        shortnames = [''.join(c for c in s if c.isupper()) for s in basenames]
        if len(shortnames) != len(set(shortnames)):
            raise ValueError('Please distinguish folders in processed data '
                             'folder by having unique combinations of '
                             'uppercase letters in their names.')
        return dict(zip(shortnames, basenames))
    
    def _get_rat_overview(self):
        '''
        Load the Rat_Overview file and remove the entries without data.

        Returns
        -------
        pd.DataFrame

        '''
        try:
            return pd.read_excel(
                os.path.join(os.path.dirname(self.path), 'Rat_Overview.xlsx')
                ).drop('Notes', axis=1).dropna()
        except Exception:
            print('Warning: Rat_Overview file could not be loaded!')
            return None
    
    def prepare_savedirs(self, *newdirs):
        '''
        Creates folders of the form self.savedirs/newdirs[0]/newdirs[1]...
        (for each path in self.savedirs).

        Parameters
        ----------
        *newdirs : str
            Variable number of folder names.

        Raises
        ------
        ValueError
            Requires nonzero number of string inputs.

        Returns
        -------
        None.

        '''
        basepaths = self.savedirs
        if len(newdirs) != 0:
            if all(isinstance(p, str) for p in newdirs):
                for path in basepaths:
                    path = os.path.join(path, *newdirs)
                    if not os.path.isdir(path):
                        os.makedirs(path, mode = 0o766)
            else:
                raise ValueError('prepare_savedirs inputs must be strings.')
        else:
            raise ValueError('prepare_savedirs requires at least one input.')
    
    def load_exam(self, exam_id, msg = True):
        '''
        Loads a single exam into self.exam.
        self.exam contains BrPyLoader object.
        self.current_exam_id contains that object's exam_id
        
        Parameters
        ----------
        exam_id : int or str
            ID of the exam we wish to load
        msg : bool, optional
            Toggles printing of available scans and recos. The default is True.

        Returns
        -------
        None.

        '''
        for path in self.paths_loaded:
            if str(exam_id) in path:
                self.exam = load(path, msg = msg)
                self.current_exam_id = exam_id
                
    def load_exams(self, msg = False):
        '''
        Loads all available exams into self.avail_exams.
        Each exam_id contains BrPyLoader object and a dictionary showing the 
        scan_id and reco_id structure.

        Parameters
        ----------
        msg : bool, optional
            Toggles printing of available scans and recos. The default is True.

        Returns
        -------
        None.

        '''
        for path, exam_id in zip(self.paths_loaded, self.avail_exam_ids):
            temp_loaded = load(path, msg = False)
            self.avail_exams[exam_id] = [temp_loaded, 
                                         temp_loaded.available_scans]
        if msg:
            pprint(self.avail_exams)
    
    def pull_exam_data(self, exam_id, scan_id = None, reco_id = None, 
                       as_numpy = True, clear = False, to_self = False):
        '''
        Method for accessing exam data. Behaviour changes based on input:
            *exam_id is required for all. scan_id required when reco_id passed* 
            
            exam_id only - creates self.exam_data containing a dict with all 
                           scans and recos.
            + scan_id - creates self.scan_data containing a dict with recos for
                        a particular scan. Contains metadata too.
            + reco_id - creates self.reco_data containing a dict with one reco.
                        Contains metadata too.
            
            exam_id == 'exam' - loads scans from exam loaded with load_exam().

        Parameters
        ----------
        exam_id : int or str
        
        scan_id : int, optional
        
        reco_id : int, optional
        
        as_numpy : bool, optional
            Returns numpy array, otherwise Nifti1Image. The default is True.
        clear : bool, optional
            Clears the storage variable (self.exam_data or self.scan_data or 
            self.reco_data) before appending a new value. The default is False.
        to_self : bool, optional
            Whether to return a value or just add it to a container in self.
            The default is False.

        Raises
        ------
        ValueError
            If reco_id is passed without scan_id or exam_id is not found.
        AttributeError
            load_exams() must be used before this method.

        Returns
        -------
        None.

        '''
        if not isinstance(as_numpy, bool):
            raise ValueError('as_numpy must be boolean.')
        if not isinstance(clear, bool):
            raise ValueError('clear must be boolean.')
        exam_id = str(exam_id)
        
        if exam_id.isdecimal():
            if self.avail_exams:
                if exam_id in self.avail_exams:
                    ex = self.avail_exams[exam_id][0]
                    #Gets all recos under all scans for an exam. Currently 
                    #doesn't include the acqp, method etc. files.
                    if scan_id is None and reco_id is None:
                        if clear:
                            self.exam_data = []
                        output = {exam_id : ex.get_all_scans(as_numpy)}
                        if not to_self:
                            return output
                        else:
                            self.exam_data = output
                    #Gets all recos under given scan.
                    elif isinstance(scan_id, self.int) and reco_id is None:
                        if clear:
                            self.scan_data = []
                        output = {
                            "exam_id" : exam_id,
                            "scan_id" : scan_id,
                            "acqp"    : ex.get_acqp(scan_id),
                            "method"  : ex.get_method(scan_id),
                            "recos"   : ex.get_all_recos(scan_id)
                            }
                        if not to_self:
                            return output
                        else:
                            self.scan_data.append(output)
                    #Gets a particular reco under a particular scan.
                    elif isinstance(scan_id, self.int) and isinstance(reco_id, self.int):
                        if clear:
                            self.reco_data = []
                        if as_numpy:
                            names = ('data', 'affine', 'header', 'visu_pars')
                        else:
                            names = ('data', 'visu_pars')
                        output = {
                            "exam_id" : exam_id,
                            "scan_id" : scan_id,
                            "acqp"    : ex.get_acqp(scan_id),
                            "method"  : ex.get_method(scan_id),
                            "recos"   : dict(
                                zip(
                                    names, 
                                    ex.get_dataobj(scan_id, reco_id, as_numpy) 
                                    + (ex.get_visu_pars(scan_id, reco_id), )
                                    )
                                )
                            }
                        if not to_self:
                            return output
                        else:
                            self.reco_data.append(output)
                    else:
                        raise ValueError(f'Unsupported input: {exam_id=}, '
                                         f'{scan_id=}, {reco_id=}.')
                else:
                    raise ValueError(f'Exam id {exam_id} was not found.')
            else:
                raise AttributeError('No exams had been loaded. Please use '
                                     'load_exams() first if id is int.')
                
        else:
            if self.exam:
                if clear:
                    self.exam_data = []
                output = self.exam.get_all_scans(as_numpy)
                if not to_self:
                    return output
                else:
                    self.exam_data = output
            else:
                raise AttributeError('No exam had been loaded. Please use '
                                     'load_exam() first if exam_id = "exam".')
    
    def show_attributes(self):
        return vars(self)
    
    def pull_processed_data(self, exam_id, dir_label, to_dict = True, 
                            as_numpy = True, substring = ''):
        '''
        A method for fetching saved processed data. Returns each dataset in
        the order it appears in file explorer.

        Parameters
        ----------
        exam_id : int/str
            Exam ID which gets converted into a string anyway.
        dir_label : str/list[str]
            Label of the folder containing the processed data. Few options:
            1. Use just the shortcut of the folder name (all capital letters). 
               Full list found in self.processed_dirs.
            2. Use full folder name.
            3. Use a list of full folder names.
        to_dict : bool, optional
            Returns the path as key if True. The default is True.
        as_numpy : bool, optional
            Returns a numpy array otherwise Nifti1Image. The default is True.
        substring : str, optional
            A substring in filepath to be found. The default is ''.

        Raises
        ------
        ValueError
            Exam ID must be a valid integer.
        FileNotFoundError
            The folder with processed data is empty.
        ImageFileError
            glob() likely returned non-image file or folder, use substring=
        Returns
        -------
        list[np.ndarray/Nifti1Image] / dict[str : np.ndarray/Nifti1Image]
            Depends on to_dict and [as_numpy].

        '''
        
        self.processing_id = exam_id
        exam_id = str(exam_id)
        if not exam_id.isdecimal():
            raise ValueError('Exam id must be a number.')
        #stops searching the list once it finds the path to the exam
        exam_path = next((p for p in self.savedirs if exam_id in p), None)
        self.processing_path = exam_path
        #input checks
        if isinstance(dir_label, str):
            if dir_label.isupper():
                dirpath = self.processed_dirs[dir_label]
            else:
                dirpath = dir_label
        elif isinstance(dir_label, list) and all(isinstance(x,str) for x in dir_label):
            dirpath = os.path.join(*dir_label)
        else:
            raise ValueError('dir_label must be either string or list of strings.')
        #when recursive=False (default) using /** should be the same as /*
        data_paths = glob(f'{os.path.join(exam_path, dirpath)}/*{substring}*')
        if not data_paths:
            raise FileNotFoundError('Failed to find path to data.')
        if as_numpy:
            output = [nib.load(data_path).get_fdata() 
                      for data_path in data_paths]
        else:
            output = [nib.load(data_path) for data_path in data_paths]
        if to_dict:
            return dict(
                zip(
                    [os.path.basename(data_path) for data_path in data_paths],
                    output
                    )
                )
        else:
            return output
    
    def save_processed_data(self, newdata, dirname, filename, affine, 
                            header=None, msg=True, processing_id=None,
                            subfolders=None, overwrite=True, ext='.nii.gz'):
        '''
        Saves data of the exam currently being processed (from 
        self.processing_id) to a nifti file in folder 'dirname'.

        Parameters
        ----------
        newdata : np.ndarray
        dirname : str
            Shortcut of the folder to store the new data.
            The shortcut is made of all uppercase letters in the folder name.
        filename : str
            The file will be named {self.processing_id}_{filename}.nii.gz
        affine : np.ndarray
        header : nibabel.nifti1.Nifti1Header or None, optional
            The default is None.
        msg : bool, optional
            Lets you know which exam id is being used. The default is True.
        processing_id : int, optional
        subfolders : list[string], optional
            Choose a path to a subfolder to save the data to.
        overwrite : bool, optional
            Whether or not to overwrite existing files.

        Raises
        ------
        ValueError
            Use self.pull_processed_data() first.

        Returns
        -------
        None.

        '''
        if self.processing_id is None and processing_id is None:
            raise ValueError('No exam being processed at the moment.')
        if subfolders is not None:
            if not isinstance(subfolders, list):
                raise ValueError('subfolders must be a list or None')
            elif not all(isinstance(x, str) for x in subfolders):
                raise ValueError('subfolders must contain only strings')
        if processing_id is None:
            processing_id = self.processing_id
        if msg:
            print(f'Saving exam {processing_id}.')
        exampath = next((p for p in self.savedirs if str(processing_id) in p), None)
        dirpath = os.path.join(exampath, self.processed_dirs[dirname])
        if subfolders is not None:
            dirpath = os.path.join(dirpath, *subfolders)
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(f'Could not find {dirpath} when saving '
                                    f'exam {processing_id}.')
        filepath = os.path.join(dirpath, f'{processing_id}_{filename}{ext}')
        if not overwrite and os.path.isfile(filepath):
            return
        nifti = nib.Nifti1Image(newdata, affine, header=header)
        nib.save(nifti, filepath)
    
    def gen_processed_data(self, dir_short, substring):
        '''
        Yields processed data specified by dir_short and substring.

        Parameters
        ----------
        dir_short : str
            Shortcut of a directory made of capital letters in its name.
        substring : str
            Substring that identifies all files we want to return.

        Yields
        ------
        nifti : nib.Nifti1Image
        arr : numpy.ndarray
        name : str
            File name of the nifti file.
        exam_id : str

        '''
        for exam_id in self.avail_exams:
            #dictionary with images
            imgdict = self.pull_processed_data(
                exam_id, 
                dir_short,
                to_dict = True,
                as_numpy = False,
                substring = substring
                )
            if not imgdict:
                print(f'Exam {exam_id} does not contain the processed data '
                      f'using {dir_short=} and {substring=}')
                continue
            #for path, nifti in mag_nifti:
            for name, nifti in imgdict.items():
                arr = nifti.get_fdata()
                yield nifti, arr, name, exam_id
    
    def save_bval_bvec(self):
        '''
        Loads effective b-values and b-vectors from the method file and saves 
        them in BVals folder in processed data folder. Only the first 5 decimal
        places of each datapoint are saved.

        Returns
        -------
        None.

        '''
        rat_overview = self.rat_overview
        #extract the columns with phase contrast study_ids
        phasecon_cols = [col for col in rat_overview.columns if 'phase contrast' in col.lower()]
        phasecon_studies = rat_overview[phasecon_cols]
        _zip = (
            [x[0] for x in self.avail_exams.values()], 
            self.savedirs, 
            phasecon_studies.astype(int).values,
            )
        self.prepare_savedirs('BVals')
        #Loader is bruker_to_py.BrPyLoader
        for Loader, savedir, study_ids in zip(*_zip):
            for (study_id, study_label) in zip(study_ids, phasecon_studies.columns):
                study_label = study_label.replace(' ', '_')
                method = Loader.get_method(study_id)
                bval = np.array(method['PVM_DwEffBval'])
                bvec = np.array(method['PVM_DwGradVec'])
                #transpose to write the array's columns row-wise
                np.savetxt(
                    os.path.join(savedir, 'BVals', f'{study_label}.bval'), 
                    #transposing a 1D array
                    bval.reshape(1, bval.shape[0]),
                    fmt = '%.5f'
                    )
                np.savetxt(
                    os.path.join(savedir, 'BVals', f'{study_label}.bvec'), 
                    bvec.T,
                    fmt = '%.5f'
                    )
    
    def get_processed_metadata(self, scan_id, filename):
        '''
        Gets method or acqp file of study = study_id for the last loaded 
        processed exam. 
        For example:
        >>> self.pull_processed_data(exam_id)
        -> self.processed_id contains exam_id
        >>> self.get_processed_metadata(study_id, 'method')
        -> retrieves the method.npy for study = study_id from exam = exam_id

        Parameters
        ----------
        scan_id : int
            ID number of the scan from which the metadata is pulled.
        filename : str
            'method', 'acqp' or 'both'.

        Raises
        ------
        ValueError
            self.processing_id needs to contain a number.
            study_id must be an integer.
            filename must be either 'method', 'acqp' or 'both'.

        Returns
        -------
        OrderedDict
            method if filename is 'method'
        OrderedDict
            acqp if filename is 'acqp'
        tuple(OrderedDict, OrderedDict)
            (method, acqp) if filename is 'both'

        '''
        #check input
        if self.processing_id is None:
            raise ValueError(
                'First load an exam using pull_processed_data() before using '
                'get_processed_metadata()'
                )
        if not isinstance(filename, str):
            raise ValueError('filename must be a string')
        if not isinstance(scan_id, self.int):
            raise ValueError('scan_id must be an integer')
        filename = filename.lower()
        if filename not in ('method', 'acqp', 'both'):
            raise ValueError('filename options are either "method", "acqp" or '
                             'both')
        
        current_exam_id = str(self.processing_id)
        print(f'Retrieving {filename} file for exam {current_exam_id}.')
        loader = self.avail_exams[current_exam_id][0]
        if filename == 'method':
            return loader.get_method(scan_id)
        elif filename == 'acqp':
            return loader.get_acqp(scan_id)
        else:
            return loader.get_method(scan_id), loader.get_acqp(scan_id)

# def load_bruker(path, fix_DWI_shape = True, endswith = '_1_1'):
#     'BrkRaw does not with in python 3.9.15 for some reason.'
#     import brkraw
#     #C:/Users/mbcxamk2/OneDrive - The University of Manchester/Uni/PhD project/Rat_studies/Rat_MRI_data
#     paths = glob(f'{path}/*{endswith}')
#     for drctry in paths:
#         #chr(92) is a backslash but f-strings don't allow them yet
#         print(f'Loading {drctry.split(chr(92))[-1]}')
#         #brkraw takes care of the complexity of loading and verifying exams
#         pvdset = brkraw.load(drctry)
#         #dictionary of available scans of the form
#         # {scan_id:[reco_id, reco_id...],...}
#         avail = pvdset._avail
#         #folder structure: {drctry}/{scan_id}/pdata/{reco_id}
#         #there is a bunch of stuff inside scan_id and reco_id, only pulling out a few things
#         for scan_id, reco_ids in avail.items():
#             #create a new path to save the loaded stuff to
#             newdir = _make_dirs(f"{drctry}_loaded/{scan_id}")
#             #.parameters retreives the OrderedDict from the objects
#             acqp = pvdset.get_acqp(scan_id).parameters
#             method = pvdset.get_method(scan_id).parameters
#             np.save(f"{newdir}/acqp.npy", acqp)
#             np.save(f"{newdir}/method.npy", method)
#             for reco_id in reco_ids:
#                 newrecodir = os.path.join(str(newdir), 'pdata', str(reco_id))
#                 filename = f'niiobj_{reco_id}'
#                 ext = 'nii.gz'
#                 pvdset.save_as(scan_id, reco_id, filename, 
#                                dir=_make_dirs(newrecodir), ext=ext)
#                 visu_pars = pvdset.get_visu_pars(scan_id, reco_id)
#                 np.save(f"{newrecodir}/visu_pars.npy", visu_pars.parameters)
#                 if fix_DWI_shape:
#                     _fix_DWI_shape(f'{filename}.{ext}', newrecodir, method)
#         print('Done!\n')
#     print('-<# Finished loading all studies. #>-')

# def _make_dirs(dirpath):
#     """
#     Checks if a directory exists and if not, creates it including 
#     subdirectories. Useful when trying to save a file to a new directory.
    
#     Parameters
#     ----------
#     filepath : str
#         File path to save the file.
#     Returns
#     -------
#     filepath : str
#     """
    
#     if not os.path.isdir(dirpath):
#         os.makedirs(dirpath, mode = 0o766)
#     return dirpath

# def _fix_DWI_shape(filename, dirpath, method):
#     if not 'DtiEpi' in method['Method']:
#         return
#     print('Reshaping data.')
#     path = os.path.join(dirpath, filename)
#     nifti = nib.load(path)
#     arr = nifti.get_fdata()
#     header = nifti.header
#     #reshape into (x, y, z, b0 + directions, repetitions)
#     arr = np.reshape(
#         arr,
#         (*arr.shape[:3],
#          method['PVM_DwAoImages'] + method['PVM_DwNDiffDir'],
#          method['PVM_NRepetitions']),
#         order='F'
#         )
#     #assign new dimensions to the header
#     header['dim'] = [
#         arr.ndim,
#         *arr.shape,
#         #['dim'] has form len([ndim, x, y, z, n or 1, n or 1, ...]) == 8
#         *[1]*(7-arr.ndim)
#         ]
#     new_nifti = nib.Nifti1Image(arr, nifti.affine, header=header)
#     nib.save(new_nifti, path)
#     print(f'Data reshaped and saved in {path}.')

if __name__ == "__main__": 
    
    test = load("C:/Users/mbcxamk2/OneDrive - The University of Manchester/Uni"
                "/PhD project/Rat_studies/Rat_MRI_data/"
                "20220907_145115_220530_1_1_loaded")
    test_reco = test.get_all_recos(1)
    