"""Module to process diffusion tensor imaging (DTI) data using DiPy.

@author: Martin Kozár (University of Manchester)
@contact: martin.kozar@manchester.ac.uk
"""

import sys
import time
import pprint
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from dipy.align import motion_correction
from dipy.align.imaffine import AffineMap
from dipy.core.gradients import GradientTable, gradient_table
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.localpca import mppca
from dipy.denoise.noise_estimate import estimate_sigma, piesno
from dipy.denoise.patch2self import patch2self
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.dti import TensorFit, TensorModel
from dipy.segment.mask import median_otsu

if sys.version_info < (3, 10):
    raise ImportError(
        "Requires at least python 3.10 for the use of union types in the type "
        "hints. If those are removed, the code should work with python 3.8+ "
        "(untested)."
        )

#make mpire optional
WorkerPool: Any = None
DiPyPathHandler: Any = None
try:
    from mpire import WorkerPool
    from path_handler import DiPyPathHandler, init
except ImportError:
    print("mpire and path_handler not found, multiprocessing pipeline will not "
          "work.")

class DiPyDTI():
    """Class to handle diffusion tensor imaging (DTI) processing with DiPy.
    User should mainly interact with the load_data and run_pipeline methods. 
    5D data (x,y,z,diffusion,repetition) is reshaped to 4D for motion 
    correction and repetitions are averaged before the other steps. bvals and 
    bvecs are automatically prepared for multiple repetitions by repeating them 
    num_reps times.

    Definitions of ID numbers (following brkraw naming convention):
        - Exam ID: The MR number, number of a new exam card in paravision, 
        e.g. 230215.
        - Scan ID: The number of the scan, e.g. 10, shown in paravision as E10 
        on the exam card.
        - Reco ID: The reconstruction number, e.g. 1. The default image is 1, 
        any additional processing, such as returning phase images, will have
        a different reco id. Diffusion images are usually reco id 1.

    Example usage:
    --------------
    from dipy_dti_fit import DiPyDTI
    from path_handler import DiPyPathHandler, init

    # Load the data (using DiPyPathHandler)
    path = str(Path.cwd() ... / 'MRI_data') # locate your data folder
    data_object = init(path, msg=False)
    paths_dict = DiPyPathHandler(data_object, exam_id).get_data_paths(
        dti_col=dti_col, id_col=id_col, return_metadata=True
        )
    dti = DiPyDTI(paths_dict)

    # Load the data (using load_data method)
    dti = DiPyDTI()
    dti.load_data(data_path, bvals_path, bvecs_path, mask_path, savedir)

    # Load the data (using make_paths_dict method)
    paths_dict = DiPyDTI.make_paths_dict(data_path, bvals_path, bvecs_path, 
        mask_path, savedir)
    dti = DiPyDTI(paths_dict)

    # Loading the data allows for additional kwargs to be passed in
    dti = DiPyDTI(paths_dict, exam_id=230215, scan_id=10, check_data=False)

    # Set the save directory (if not provided in the paths_dict)
    dti.set_savedir(savedir)

    # Check the paths
    print(dti)

    # Run the pipeline (all steps are saved by default, the folder that holds
    # the saved data is determined by .savedir attribute of the DiPyDTI object)
    dti.run_pipeline(pipeline=['motion_correction', 'denoise', 'fit_dti'])

    # Full example on multiple datasets using the bruker_to_py.py module
    path = str(Path.cwd().parent / 'MRI_data') # running from Scripts folder next to MRI_data
    data_object = init(path, msg=False)
    for exam_id in data_object.avail_exam_ids:
        path_handler = DiPyPathHandler(data_object, exam_id)
        dpdti = DiPyDTI(
            path_handler.get_data_paths(
                'dti', id_col='MR #', return_metadata=True
                )
            )
        dpdti.run_pipeline(
            pipeline = ["motion_correction", "degibbs", "denoise", "fit_dti"],
            kwargs = {'degibbs':{'num_processes':3}}
            )

    Explanation of the main methods:
    --------------------------------
    Loading data:
        - The load_data method takes in paths to the diffusion data, bvals, 
        bvecs, and mask (if available) and loads them into the DiPyDTI object. 
        These paths can also be provided as an input dictionary to the DiPyDTI 
        object.
        - Alternatively, the DiPyPathHandler class can be used to find the scan 
        id and get the paths, but it expects a DataObject instance from 
        bruker_to_py.py. It can also return the methods, acqp, and visu_pars
        dictionaries. Method is used to extract the number of repetitions, which
        can be set manually using the num_reps attribute.
        - If you have the paths, you can use the DiPyDTI.make_paths_dict method 
        to create the dictionary of paths.

    Loading data for running individual methods manually:
        - Replace the data_path in the paths_dict with the path to the saved
        data from the previous step and you can run the individual methods.
        - If you've run some of the processing steps and your repetitions were 
        averaged out, you need to be careful since loading the methods file will
        give you the original number of repetitions. You can set the num_reps
        attribute manually to the number of repetitions you have.
        - The checking of the loaded data can be turned off by setting 
        check_data=False in the load_data method.
        - bvals and bvecs repetitions can be turned off by setting 
        prepare_b0s=False in the load_data method.
        - exam_id and scan_id can be set manually as kwargs to either the 
        load_data method or the DiPyDTI object.

    Saving data:
        - Steps in the pipeline are saved by default, but you can also save
        individual data arrays using the save method. A DiPyDTI folder is 
        created in the savedir directory and each processing step is saved in a 
        subfolder.

    Masking (optional):
        - Options to either have the pipeline extract a brain mask using the 
        option 'extract_brain' or to provide a path to a mask file, or if 
        neither is selected, then the dti fit will be done on the whole brain.

    Running the pipeline:
        - The run_pipeline method takes in a list of processing steps to run and
        a dictionary of additional arguments to pass into each step. The steps
        are: 'motion_correction', 'degibbs', 'denoise', 'extract_brain', 
        'fit_dti'.
        - The kwargs dictionary should have keys corresponding to the steps in 
        the pipeline and the values should be dictionaries of additional 
        arguments to pass into each step.

    Default pipeline changes to DiPy default values (and original defaults):
        - The motion_correction step uses the pipeline of 
        ['center_of_mass', 'translation', 'rigid'] (no 'affine').
        - The median_otsu function used in extract_brain uses the default
        parameters of median_radius=2, numpass=1 (4, 4).

    Default choices for algorithm selection:
        - The denoise step uses the method 'mppca'.
        - The fit_dti step uses the model 'WLS'.
        - The estimate_noise function used in RESTORE model uses the estimator 
        'piesno'.

    Attributes:
    -----------
    data_path : Path
        Path to the diffusion data.
    bvals_path : Path
        Path to the bvals file.
    bvecs_path : Path
        Path to the bvecs file.
    mask_path : Path
        Path to the mask file.
    savedir : Path
        Path to the directory to save the processed data.
    logfile : Path
        Log file for the pipeline.
    diffusion_data : nib.Nifti1Image
        The diffusion data.
    bvals : np.ndarray
        The bvals.
    bvecs : np.ndarray
        The bvecs.
    mask : np.ndarray
        The mask.
    gtab : GradientTable
        The gradient table.
    pipeline_steps : list
        The processing steps to run.
    valid_steps : list[str]
        The valid processing steps.
    methods : OrderedDict
        The method dictionary from the bruker data.
    acqp : OrderedDict
        The acqp dictionary from the bruker data.
    visu_pars : OrderedDict
        The visu_pars dictionary from the bruker data.
    exam_id : int
        The exam id, e.g. 230215.
    scan_id : int
        The scan id, e.g. 10.
    num_reps : int
        Number of repetitions.
    motion_corrected : nib.Nifti1Image
        The motion corrected data.
    averaged_data : nib.Nifti1Image
        The data averaged across repetitions.
    reg_affines : np.ndarray
        The registration affines.
    b0_masked : nib.Nifti1Image
        The masked b0 volume.
    noise_mask : np.ndarray
        The noise mask.
    sigma : np.ndarray
        The noise estimate.
    gibbs_suppressed : nib.Nifti1Image
        The Gibbs suppressed data.
    denoised_data : nib.Nifti1Image
        The denoised data.
    model : TensorModel
        The tensor model.
    dti_fit : TensorFit
        The final DTI fit.

    Methods:
    --------
    run_pipeline(pipeline: list = ["motion_correction", "denoise", "fit_dti"],
                    kwargs: dict[dict] = {}, store_tensors: bool = False
                    ) -> None|TensorFit
        Run a pipeline of processing steps. The default pipeline is in
        self.valid_steps (defined in __init__). After motion correction, the 
        data is averaged if there are multiple repetitions.

    load_data(data_path: str, bvals_path: str, bvecs_path: str,
                mask_path: str = "", **kwargs)
        Load the data, bvals, bvecs, and mask (if provided) into the DiPyDTI 
        object. Optionally set the savedir, method, acqp, and visu_pars. If 
        method file is provided, the number of repetitions will be extracted 
        from it.

    make_gradient_table(atol: float = 1.0, **kwargs) -> GradientTable
        Create a gradient table from the bvals and bvecs.

    set_savedir(savedir: str)
        Set the save directory for the processed data.

    get_scan_ids(data_path: Path) -> tuple[int, int]
        Extract the exam and scan IDs from the data path.

    correct_motion(nifti: nib.Nifti1Image, pipeline: list = None,
                    affine: np.ndarray = None, save: bool = True
                    ) -> tuple[nib.Nifti1Image, np.ndarray]
        Correct for between-volume motion artifacts in the diffusion data.

    average_repetitions(nifti: nib.Nifti1Image, num_reps: int = 1,
                        save: bool = True) -> nib.Nifti1Image
        Average the repetitions in the diffusion data. The number of repetitions
        is extracted from the method file, or can be set manually using the 
        num_reps attribute.

    estimate_noise(nifti: nib.Nifti1Image, receiver_coils: int = 1, 
                    method: str = 'piesno', **kwargs
                    ) -> tuple[np.ndarray, np.ndarray]
        Estimate the noise from the diffusion data. noise_mask is only returned 
        for piesno and it's a mask of the background where the noise is 
        calculated from.

    degibbs(nifti: nib.Nifti1Image, slice_axis = 2, save: bool = True, **kwargs)
        Remove Gibbs ringing artifacts from the diffusion data. Option to use
        multiple cores by specifying num_processes=. Warning: do not set the 
        inplace= argument to True.

    denoise(nifti: nib.Nifti1Image, method: str = 'mppca', save: bool = True,
            **kwargs) -> nib.Nifti1Image
        Denoise the diffusion data. Main params mentioned below, for more 
        options see the docstrings of the denoising functions.

    extract_brain(nifti: nib.Nifti1Image, save: bool = True, 
                    b0_threshold: int = 50, **kwargs
                    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]
        Extract the brain from the diffusion data.

    extract_b0vol(nifti: nib.Nifti1Image, b0_threshold: int = 50) -> np.ndarray
        Extract the b0 volume from the diffusion data.

    fit_dti(nifti: nib.Nifti1Image, save: bool = True, model: str = 'WLS',
            save_tensor: bool = False, **kwargs) -> TensorFit
        Fit the diffusion tensor model to the data. Options: 'WLS', 'RESTORE'.
        kwargs are passed into estimate_noise for the RESTORE model.

    save(data: np.ndarray|nib.Nifti1Image, newdir: str = '', filename: str = '')
        Save the data to the savedir directory.

    @staticmethod
    to_nifti(nifti: nib.Nifti1Image, data: np.ndarray) -> nib.Nifti1Image
        Convert a numpy array to a nibabel Nifti1Image.

    @staticmethod
    make_paths_dict(data_path: str, bvals_path: str, bvecs_path: str,
                    mask_path: str, savedir: str) -> dict
        Create a dictionary of paths for the DiPyDTI object.
    """
    def __init__(self, paths_dict: dict = None, **kwargs) -> None:
        # Paths
        self.data_path: Path = ""
        self.bvals_path: Path = ""
        self.bvecs_path: Path = ""
        self.mask_path: Path = ""
        self.savedir: Path = ""
        self.logfile: Path = "" # Log file for the pipeline
        # DTI inputs
        self.diffusion_data: nib.Nifti1Image = None
        self.bvals: np.ndarray = None
        self.bvecs: np.ndarray = None
        self.mask: np.ndarray = None
        self.gtab: GradientTable = None
        self.pipeline_steps: list = None
        self.valid_steps: list[str] = [
            "motion_correction", "degibbs", "denoise","extract_brain", "fit_dti"
            ]
        # Metadata
        self.method: OrderedDict = None
        self.acqp: OrderedDict = None
        self.visu_pars: OrderedDict = None
        # Scan info
        self.exam_id: int = None
        self.scan_id: int = None
        self.num_reps: int = 1 # Number of repetitions
        # Processing outputs
        self.motion_corrected: nib.Nifti1Image = None
        self.averaged_data: nib.Nifti1Image = None
        self.reg_affines: np.ndarray = None
        self.b0_masked: nib.Nifti1Image = None
        self.noise_mask: np.ndarray = None
        self.sigma: np.ndarray = None
        self.gibbs_suppressed: nib.Nifti1Image = None
        self.denoised_data: nib.Nifti1Image = None
        self.model: TensorModel = None
        self.dti_fit: TensorFit = None
        if paths_dict is not None:
            self.load_data(**paths_dict, **kwargs)

    def __str__(self) -> str:
        return f"""Paths:
-----
Data: {self.data_path}
Bvals: {self.bvals_path}
Bvecs: {self.bvecs_path}
Mask: {self.mask_path}
Savedir: {self.savedir}"""

    def __repr__(self) -> str:
        paths_dict = DiPyDTI.make_paths_dict(
            self.data_path, self.bvals_path, self.bvecs_path,
            self.mask_path, self.savedir
            )
        return f"DiPyDTI(paths_dict={paths_dict})"

    def load_data(self, data_path: str, bvals_path: str, bvecs_path: str,
                  mask_path: str = "", num_reps: int = 0,
                  prepare_b0s: bool = True, check_data: bool = True, **kwargs):
        """Load the data, bvals, bvecs, and mask (if provided) into the DiPyDTI 
        object. Optionally set the savedir, method, acqp, and visu_pars. If 
        method file is provided, the number of repetitions will be extracted 
        from it. exam_id and scan_id can be set manually as kwargs, otherwise
        they will be extracted from the data path.

        Parameters
        ----------
        data_path : str
            Path to the diffusion data. Expecting a 4D or 5D nifti file of shape
            (x, y, slices, n_diffusion_volumes, (n_repetitions)).
        bvals_path : str
            Path to the bvals file. Expecting a 1D numpy array.
        bvecs_path : str
            Path to the bvecs file. Expecting a 2D numpy array.
        mask_path : str, optional
            Path to the mask file, by default "".
        savedir : str, optional
            Path to the directory to save the processed data, by default "".
        method, acqp, visu_pars : OrderedDict, optional
            The method, acqp, and visu_pars dictionaries from the bruker data,
            by default None.
        num_reps : int, optional
            Number of repetitions in the diffusion data, by default 1.
        prepare_b0s : bool, optional
            Whether to prepare the bvals and bvecs by repeating them num_reps 
            times (need as many values as there are diffusion volumes), by 
            default True.
        check_data : bool, optional
            Whether to check the loaded data, by default True.
            
        Raises
        ------
        ValueError
            If diffusion data is not a nifti image.
            If diffusion data is not 4D or 5D.
            If bvals is not a 1D numpy array.
            If bvals length does not match the number of diffusion volumes.
            If bvecs is not a 2D numpy array.
            If bvecs shape does not match the number of diffusion volumes.
            If bvecs does not have 3 columns.
            If mask is not a 3D numpy array.
            If mask shape does not match the diffusion data.
        
        """
        self.data_path = Path(data_path)
        self.get_scan_ids(
            self.data_path, kwargs.get('exam_id'), kwargs.get('scan_id')
            )
        self.bvals_path = Path(bvals_path)
        self.bvecs_path = Path(bvecs_path)
        self.mask_path = Path(mask_path)
        _, _, self.diffusion_data = load_nifti(data_path, return_img=True)
        self.bvals, self.bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
        if mask_path:
            self.mask, _ = load_nifti(mask_path)
        if kwargs:
            self.set_savedir(kwargs.get('savedir'))
            self.method = kwargs.get('method')
            self.acqp = kwargs.get('acqp')
            self.visu_pars = kwargs.get('visu_pars')
        self.__set_num_reps(num_reps)
        if self.num_reps != 1 and prepare_b0s:
            self.bvals = np.tile(self.bvals, self.num_reps)
            self.bvecs = np.tile(self.bvecs, (self.num_reps, 1))
        if check_data:
            self.__check_loaded_data()

    def run_pipeline(self, pipeline: list = None, kwargs: dict[dict] = None
                     ) -> None|TensorFit:
        """Run a pipeline of processing steps. The default pipeline is
        in self.valid_steps (defined in __init__). After motion correction,
        the data is averaged if there are multiple repetitions.
        
        Parameters
        ----------
        pipeline : list, optional
            The processing steps to run, 
            by default ["motion_correction", "denoise", "fit_dti"]
        kwargs : dict[dict], optional
            kwargs are passed into the individual processing steps. kwargs 
            should be a dictionary with keys corresponding to the steps in the 
            pipeline and values should be dictionaries of additional arguments 
            to pass into each step, by default {}
            
        Raises
        ------
        ValueError
            If a step in the pipeline is not implemented.
            If savedir is not set.
            If pipeline is not a list.
            If the pipeline is empty.
            If a step in pipeline is not in self.valid_steps.
            If extract_brain is in the pipeline and mask is provided.
            If a key in kwargs is not in the pipeline.
            If a value in kwargs is not a dictionary.
        
        """
        kwargs = kwargs or {}
        pipeline = pipeline or ["motion_correction", "denoise", "fit_dti"]
        self.__check_pipeline_inputs(pipeline, kwargs)
        self.pipeline_steps = pipeline.copy()
        self.__log_pipeline(pipeline, kwargs)
        current_data = self.diffusion_data
        try:
            for i, step in enumerate(pipeline):
                self.__print_step(step, i, len(pipeline))
                start = time.perf_counter()
                if step == "motion_correction":
                    correct_motion_kwargs = kwargs.get('motion_correction', {})
                    current_data, _ = self.correct_motion(
                        current_data, **correct_motion_kwargs
                        )
                if self.num_reps != 1:
                    current_data = self.average_repetitions(
                        current_data, self.num_reps
                        )
                if step == "degibbs":
                    degibbs_kwargs = kwargs.get('degibbs', {})
                    current_data = self.degibbs(current_data, **degibbs_kwargs)
                elif step == "denoise":
                    denoise_kwargs = kwargs.get('denoise', {})
                    current_data = self.denoise(current_data, **denoise_kwargs)
                elif step == "extract_brain":
                    extract_brain_kwargs = kwargs.get('extract_brain', {})
                    self.extract_brain(current_data, **extract_brain_kwargs)
                elif step == "fit_dti":
                    dti_fit_kwargs = kwargs.get('fit_dti', {})
                    self.fit_dti(current_data, **dti_fit_kwargs)
                    self.__log_step(step, t=time.perf_counter()-start)
                    return self.dti_fit
                elif step == "motion_correction":
                    pass
                else:
                    raise ValueError(f"Step {step} not recognized in the "
                                     "pipeline.")
                self.__log_step(step, t=time.perf_counter()-start)
        except Exception as e:
            print(traceback.format_exc())
            self.__log_step(step, error=e)

    def make_gradient_table(self, atol: float = 1.0, **kwargs) -> GradientTable:
        "Create a gradient table from the bvals and bvecs."
        if self.bvals is None or self.bvecs is None:
            raise ValueError("No bvals or bvecs found. Please load them first.")
        return gradient_table(self.bvals, self.bvecs, atol=atol, **kwargs)

    def set_savedir(self, savedir: str):
        "Set the save directory for the processed data."
        try:
            self.savedir = Path(savedir)
            assert self.savedir.exists()
            self.savedir = self.savedir / 'DiPyDTI'
            self.savedir.mkdir(exist_ok=True)
        except AssertionError as e:
            print(f"{savedir} is not a valid path")
            print(e)

    def get_scan_ids(self, data_path: Path, exam_id: int = None,
                     scan_id: int = None) -> tuple[int, int]:
        "Extract the exam and scan IDs from the data path."
        if exam_id is not None and scan_id is not None:
            self.exam_id = exam_id
            self.scan_id = scan_id
            return self.exam_id, self.scan_id
        try:
            self.exam_id = int(data_path.parts[-5].split('_')[2])
            self.scan_id = int(data_path.parts[-4])
            return self.exam_id, self.scan_id
        except (ValueError, IndexError) as e:
            print(f"Could not extract the exam and scan IDs from {data_path}")
            print("Please set them manually.")
            print(e)

    def correct_motion(self, nifti: nib.Nifti1Image, pipeline: list = None,
        reg_affine: np.ndarray = None, static: nib.Nifti1Image = None, 
        save: bool = True) -> tuple[nib.Nifti1Image, np.ndarray]:
        """Correct for between-volume motion artifacts in the diffusion data. 
        By default, the pipeline is ['center_of_mass', 'translation', 'rigid'].
        To apply a registration affine from a previous motion correction, supply
        both reg_affine and static (image). reg_affine is the affine output of 
        motion_correct and static is the static image used in the registration. 
        static can also be a tuple of (static_img.shape, static_img.affine). 
        static can also be the same as nifti if no rescaling is needed.
        """
        pipeline = pipeline or ["center_of_mass", "translation", "rigid"]
        self.__check_gtab()
        if nifti.ndim == 5:
            nifti = self.__reshape_5d(nifti)
        self.motion_corrected, self.reg_affines = self.__motion_correction(
            nifti, self.gtab, pipeline = pipeline, affine = reg_affine,
            static = static
            )
        if save:
            self.save(
                self.motion_corrected,
                newdir='MotionCorrected',
                filename='motion_corrected'
                )
            self.save(
                self.reg_affines,
                newdir='MotionCorrected',
                filename='motion_affines'
                )
        return self.motion_corrected, self.reg_affines

    def average_repetitions(self, nifti: nib.Nifti1Image, num_reps: int = 1,
                            save: bool = True) -> nib.Nifti1Image:
        """Average the repetitions in the diffusion data. The number of 
        repetitions is extracted from the method file, or can be set manually 
        using the num_reps attribute."""
        if num_reps == 1:
            return nifti
        print(f"Averaging {num_reps} repetitions...")
        data = nifti.get_fdata()
        if data.ndim == 4:
            data = data.reshape(*data.shape[:3], -1, num_reps)
        data_averaged = data.mean(axis=-1)
        self.averaged_data = self.to_nifti(nifti, data_averaged)
        self.bvecs = self.bvecs[:data_averaged.shape[-1]]
        self.bvals = self.bvals[:data_averaged.shape[-1]]
        self.gtab = self.make_gradient_table()
        self.num_reps = 1
        if save:
            self.save(
                self.averaged_data,
                newdir='MotionCorrected',
                filename='averaged_data'
                )
        return self.averaged_data

    def estimate_noise(self, nifti: nib.Nifti1Image, receiver_coils: int = 1,
                       method: str = 'piesno', **kwargs
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the noise from the diffusion data. noise_mask is only 
        returned for piesno and it's a mask of the background where the noise is 
        calculated from."""
        if method == 'piesno':
            self.sigma, self.noise_mask = piesno(
                nifti.get_fdata(), N=receiver_coils, return_mask=True, **kwargs
                )
        elif method == 'other' or not method:
            self.sigma = estimate_sigma(
                nifti.get_fdata(), N=receiver_coils, **kwargs
                )
        else:
            raise ValueError(f"Method {method} not recognized.")
        return self.sigma, self.noise_mask

    def degibbs(self, nifti: nib.Nifti1Image, slice_axis = 2, save: bool = True,
                **kwargs) -> nib.Nifti1Image:
        """Remove Gibbs ringing artifacts from the diffusion data. Option to use
        multiple cores by specifying num_processes=.

        Parameters
        ----------
        nifti : nib.Nifti1Image
            The diffusion data to remove Gibbs ringing from.
        slice_axis : int, optional
            The axis along which to remove Gibbs ringing, by default 2.
        save : bool, optional
            Whether to save the resulting nifti file, by default True
        """
        if kwargs.get('inplace', False) is True:
            raise ValueError("Do not set inplace=True in degibbs.")
        self.gibbs_suppressed = self.to_nifti(
            nifti,
            gibbs_removal(
                nifti.get_fdata(),
                slice_axis=slice_axis,
                inplace=False,
                **kwargs
                )
            )
        if save:
            self.save(
                self.gibbs_suppressed,
                newdir='GibbsSuppressed',
                filename='gibbs_suppressed'
                )
        return self.gibbs_suppressed

    def denoise(self, nifti: nib.Nifti1Image, method: str = 'mppca',
                save: bool = True, **kwargs) -> nib.Nifti1Image:
        """Denoise the diffusion data. Main params mentioned below, for more 
        options see the docstrings of the denoising functions.
        
        Patch2self:
            - model: str, optional
                The model to use for denoising. Options: 'ols', 'lasso', 
                'ridge', by default 'ols'.
                !Consider using 'ridge' for large datasets.
            - shift_intensity: bool, optional
                Whether to shift the intensity of the patches, by default True.
            - clip_negative_vals: bool, optional
                Whether to clip negative values in the denoised data, by default 
                False.
            - b0_threshold: int, optional
                Upper threshold for the b0 values in bvals file, by default 50.
        
        MPPCA:
            - patch_radius: int, optional
                The radius of the patch, by default 2 which corresponds to 5x5x5
                sliding window.

        Parameters
        ----------
        method : str, optional
            The denoising method to use. Options: 'patch2self', 'mppca', by 
            default 'mppca'.
        """
        if method == 'patch2self':
            self.denoised_data = self.__patch2self(nifti, self.bvals, **kwargs)
        elif method == 'mppca':
            self.denoised_data = self.__mppca(nifti, **kwargs)
        else:
            raise ValueError(f"Method {method} not recognized.")
        if save:
            self.save(
                self.denoised_data,
                newdir='Denoised',
                filename=f'denoised_{method}'
                )
        return self.denoised_data

    def extract_brain(self, nifti: nib.Nifti1Image, save: bool = True,
                      b0_threshold: int = 50, **kwargs
                      ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        "Extract the brain from the diffusion data."
        b0vol = self.extract_b0vol(nifti, b0_threshold=b0_threshold)
        if b0vol.ndim == 4:
            b0vol = b0vol.mean(axis=-1)
        b0_masked, mask = self.__median_otsu(b0vol, **kwargs)
        self.b0_masked = self.to_nifti(nifti, b0_masked)
        self.mask = self.to_nifti(nifti, mask)
        if save:
            self.save(
                self.b0_masked,
                newdir='BrainExtracted',
                filename='b0_masked'
                )
            self.save(self.mask, newdir='BrainMask', filename='brain_mask')
        return self.b0_masked, self.mask

    def extract_b0vol(self, nifti: nib.Nifti1Image, b0_threshold: int = 50
                      ) -> np.ndarray:
        "Extract the b0 volume from the diffusion data."
        b0_idx = self.bvals < b0_threshold
        return nifti.get_fdata()[..., b0_idx]

    def fit_dti(self, nifti: nib.Nifti1Image, save: bool = True,
                model: str = 'WLS', save_tensor: bool = False,
                **kwargs) -> TensorFit:
        """Fit the diffusion tensor model to the data. Options: 'WLS', 
        'RESTORE'. kwargs are passed into estimate_noise for the RESTORE model.
        """
        self.__check_gtab()
        masked_data = self.mask_data(nifti)
        self.model = self.__prepare_model(which=model, nifti=nifti, **kwargs)
        self.dti_fit = self.model.fit(masked_data)
        if save:
            self.save(
                self.dti_fit, newdir='DTIFit', filename='dti_fit', nifti=nifti,
                save_tensor=save_tensor
                )
        return self.dti_fit

    def mask_data(self, nifti: nib.Nifti1Image) -> np.ndarray:
        "Mask the data with the brain mask, if available."
        if self.b0_masked is not None:
            return self.b0_masked.get_fdata()
        elif self.mask is not None:
            # Numpy vectorization works in numpy-native axis order (dzyx)
            #if you do (xyz) * (xyzd) it gives an error that it can't broadcast
            #the shapes, so we swap the axes around, multiply and swap back
            return (nifti.get_fdata().T * self.mask.T).T
        else:
            return nifti.get_fdata()

    def save(self, data: nib.Nifti1Image|np.ndarray|TensorFit,
             newdir: str = "", filename: str = "",
             nifti: nib.Nifti1Image = None, save_tensor: bool = False):
        """Save the data to the savedir directory. If newdir is provided, a new
        subdirectory will be created in the savedir directory. The filename of 
        the saved data will be {exam_id}_{scan_id}_{filename}.

        Parameters
        ----------
        data : nib.Nifti1Image | np.ndarray | TensorFit
            The data to save.
        newdir : str, optional
            The name of the new subdirectory to save the data in, by default ""
        filename : str, optional
            The filename to save the data as, by default ""
        nifti : nib.Nifti1Image, optional
            A nifti object for retrieving affine and header info for saving fa 
            and md maps, by default None
        save_tensor : bool, optional
            Whether to save the tensor data as a pickled python object, by 
            default False

        Raises
        ------
        ValueError
            If the data type is not recognized for saving
        """
        prefix = f"{self.exam_id}_{self.scan_id}"
        if newdir:
            new_savedir = self.savedir / newdir
            new_savedir.mkdir(exist_ok=True)
        if isinstance(data, nib.Nifti1Image):
            nib.save(data, new_savedir / f'{prefix}_{filename}.nii.gz')
        elif isinstance(data, np.ndarray):
            np.save(new_savedir / f'{prefix}_{filename}.npy', data)
        elif isinstance(data, TensorFit):
            if nifti is None:
                raise ValueError("Nifti object required for saving dti fit.")
            nib.save(
                self.to_nifti(nifti, data.fa),
                new_savedir / f'{prefix}_fractional_anisotropy.nii.gz'
                )
            nib.save(
                self.to_nifti(nifti, data.md),
                new_savedir / f'{prefix}_mean_diffusivity.nii.gz'
                )
            if save_tensor:
                np.save(new_savedir / f'{prefix}_{filename}.npy', data)
        else:
            raise ValueError(f"Data type {type(data)} not recognized for "
                             "saving.")

    @staticmethod
    def apply_affine(moving: nib.Nifti1Image, affine: np.ndarray, 
                     static: nib.Nifti1Image|tuple[tuple, np.ndarray] = None, 
                     interpolation: str = 'nearest'
                     ) -> tuple[nib.Nifti1Image, np.ndarray]:
        """Apply an affine matrix to the moving image based on affine and the 
        static image. The static image is used to determine the shape and affine
        and can be supplied as a nibabel image or a tuple of (shape, affine). If 
        static is not provided, the moving image shape and affine are used."""
        static = static or moving
        transformed = []
        moving_data = moving.get_fdata()
        moving_shape, moving_affine = moving.shape[:-1], moving.affine
        if isinstance(static, nib.Nifti1Image):
            static_shape, static_affine = static.shape[:-1], static.affine
        else:
            static_shape, static_affine = static
        for i in range(moving.shape[-1]):
            affine_map = AffineMap(
                affine[..., i],
                static_shape,
                static_affine,
                moving_shape,
                moving_affine
            )
            transformed.append(
                affine_map.transform(
                    moving_data[..., i],
                    interpolation=interpolation
                )
            )
        return DiPyDTI.to_nifti(moving, np.stack(transformed, axis=3)), affine

    @staticmethod
    def make_paths_dict(data_path: str, bvals_path: str, bvecs_path: str,
                        mask_path: str = "", savedir: str = ""):
        """Create a dictionary of paths for easy loading of the DiPyDTI object.
        
        Args:
            Compulsory: data_path, bvals_path, bvecs_path
            Optional: mask_path, savedir
        """
        return {"data_path": data_path, "bvals_path": bvals_path,
                "bvecs_path": bvecs_path, "mask_path": mask_path, 
                "savedir": savedir}

    @staticmethod
    def to_nifti(nifti, array) -> nib.Nifti1Image:
        "Convert an array to a Nifti1Image."
        return nib.Nifti1Image(array, nifti.affine, nifti.header)

    def __set_num_reps(self, num_reps: int):
        "Set the number of repetitions."
        if not isinstance(num_reps, int):
            raise ValueError("Number of repetitions should be an integer.")
        if num_reps != 0:
            self.num_reps = num_reps
        elif self.method is not None:
            self.num_reps = self.method['PVM_NRepetitions']
        elif self.num_reps == 0:
            self.num_reps = 1

    def __check_loaded_data(self):
        "Check if the loaded data is valid."
        print("Checking loaded data...")
        # Diffusion data
        if not isinstance(self.diffusion_data, nib.Nifti1Image):
            raise ValueError("Diffusion data should be a nifti image.")
        if self.diffusion_data.ndim not in (4, 5):
            raise ValueError("Diffusion data should be 4D or 5D.")
        if self.diffusion_data.shape[-1] > self.diffusion_data.shape[0]:
            print("Last dimension of the diffusion data is larger than first.")
            print("Please double-check the data shape.")
            print("The data shape should be (x, y, slices, n_diffusion_volumes,"
                  "(n_repetitions)) when calling self.diffusion_data.shape")
        # Bvals and bvecs
        if not isinstance(self.bvals, np.ndarray) or self.bvals.ndim != 1:
            raise ValueError("BVals should be a 1D numpy array.")
        if len(self.bvals) != self.diffusion_data.shape[-1]:
            raise ValueError("Number of bvals should match the number of "
                             "volumes (diffusion directions).")
        if not isinstance(self.bvecs, np.ndarray) or self.bvecs.ndim != 2:
            raise ValueError("BVecs should be a 2D numpy array.")
        if self.bvecs.shape[1] != 3:
            raise ValueError("BVecs should have 3 columns.")
        if self.bvecs.shape[0] != self.diffusion_data.shape[-1]:
            raise ValueError("Number of bvecs should match the number of "
                             "volumes (diffusion directions).")
        # Mask
        if self.mask is not None:
            if not isinstance(self.mask, np.ndarray) or self.mask.ndim != 3:
                raise ValueError("Mask should be a 3D numpy array.")
            if self.mask.shape != self.diffusion_data.shape[:3]:
                raise ValueError("Mask shape should match the diffusion data.")
        print("Loaded data is valid.")

    def __check_pipeline_inputs(self, pipeline: list, kwargs: dict):
        "Check the inputs for the run_pipeline method."
        print("Checking pipeline inputs...")
        # Check if savedir is set
        if not self.savedir:
            raise ValueError("No save directory set. Please set one using the "
                             "set_savedir method.")
        # Verify pipeline steps
        if not isinstance(pipeline, list):
            raise ValueError("Pipeline should be a list of processing steps.")
        if not pipeline:
            raise ValueError("No pipeline steps provided.")
        for step in pipeline:
            if step not in self.valid_steps:
                raise ValueError(f"Step {step} not recognized in the pipeline.")
        if "extract_brain" in pipeline and self.mask is not None:
            raise ValueError("Cannot extract brain if a mask is provided.")
        # See if every kwargs step exists in pipeline and if each value is dict
        for step, value in kwargs.items():
            if step not in pipeline:
                raise ValueError(
                    f"Step {step} in kwargs not found in pipeline."
                    )
            if not isinstance(value, dict):
                raise ValueError(
                    f"Value for step {step} in kwargs should be a dictionary."
                    )
        print("Pipeline inputs are in valid format.")

    def __log_pipeline(self, pipeline: list, kwargs: dict):
        "Log the pipeline parameters to a text file."
        file_timestr = time.strftime("%Y%m%d-%H%M%S")
        self.savedir = self.savedir / f"{file_timestr}"
        self.savedir.mkdir(exist_ok=True)
        timestr = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"""{timestr} - Ran DiPyDTI pipeline on the following data:

{str(self)}

The pipeline steps were:
{pipeline}

Additional keyword arguments were:
{pprint.pformat(kwargs, indent = 4)}

If a step was successful, it will be logged here:"""
        self.logfile = self.savedir / f'DiPyDTI_log_{file_timestr}.txt'
        with open(self.logfile, 'w', encoding="utf-8") as f:
            f.write(log_msg)
        print(f"Logging the pipeline to {self.logfile}")
        print(log_msg)

    def __print_step(self, step: str, i: int = None, t: int = None):
        """Print the step to the console surrounded by a box, example:
        ┌────────────────────┐
        |Step 1 of 5: Denoise|
        └────────────────────┘
        
        Parameters
        ----------
        step: str
            The name of the step.
        i: int, optional
            The step number.
        t: int, optional
            The total number of steps.
        
        Raises
        ------
        ValueError
            If total steps are provided, step number should be provided as well.
        """
        if t is not None and i is None:
            raise ValueError("If total steps are provided, step number should "
                                "be provided as well.")
        step_num = f"Step {i}" if i is not None else ""
        total_steps = f" of {t}" if t is not None else ""
        connector = ": " if i is not None else ""
        step = step.replace('_', ' ').replace(' dti', ' DTI').capitalize()
        message = f"{step_num}{total_steps}{connector}{step}"
        print(f"┌{'─' * (len(message))}┐")
        print(f"|{message}|")
        print(f"└{'─' * (len(message))}┘")

    def __log_step(self, step: str, error: Exception = None,
                  t: float|str = ""):
        "Append the step to the log file."
        step = step.capitalize()
        with open(self.logfile, 'a', encoding="utf-8") as f:
            if error is None:
                f.write(f"\n{step} ran succesfully in {t:.2f} seconds.")
            else:
                f.write(f"\n{step} failed with the following error: {error}")
                f.write("\nExiting the pipeline.")

    def __motion_correction(self, nifti: nib.Nifti1Image, gtab: GradientTable,
                            pipeline: list = None, affine: np.ndarray = None,
                            static: nib.Nifti1Image = None
                            ) -> tuple[nib.Nifti1Image, np.ndarray]:
        "Run the motion correction pipeline."
        pipeline = pipeline or ["center_of_mass", "translation", "rigid"]
        if affine is not None and static is not None:
            if affine.shape[-1] != nifti.shape[-1]:
                raise ValueError("Affine matrix should have the same number of "
                                 "volumes as the diffusion data.")
            return self.apply_affine(nifti, affine, static)
        return motion_correction(nifti, gtab, pipeline=pipeline)

    def __reshape_5d(self, nifti: nib.Nifti1Image) -> nib.Nifti1Image:
        "Reshape the 5D data to 4D."
        assert nifti.ndim == 5, "Data should be 5D."
        print("Reshaping 5D data to 4D...")
        return self.to_nifti(
            nifti,
            nifti.get_fdata().reshape(*nifti.shape[:3], -1)
            )

    def __patch2self(self, nifti: nib.Nifti1Image, bvals, model='ols',
                    shift_intensity=True, clip_negative_vals=False,
                    b0_threshold=50, **kwargs) -> nib.Nifti1Image:
        "Run the patch2self denoising method with default parameters."
        return self.to_nifti(
            nifti,
            patch2self(
                nifti.get_fdata(),
                bvals,
                model=model,
                shift_intensity=shift_intensity,
                clip_negative_vals=clip_negative_vals,
                b0_threshold=b0_threshold,
                **kwargs
                )
            )

    def __mppca(self, nifti: nib.Nifti1Image, **kwargs) -> nib.Nifti1Image:
        "Run the MPPCA denoising method."
        return self.to_nifti(
            nifti,
            mppca(nifti.get_fdata(), **kwargs)
            )

    def __prepare_model(self, which: str = 'WLS', nifti: nib.Nifti1Image = None,
                       **kwargs) -> TensorModel:
        "Prepare the tensor model for fitting."
        if which == 'WLS':
            return TensorModel(self.gtab, fit_method='WLS')
        elif which == 'RESTORE':
            sigma, _ = self.estimate_noise(nifti, **kwargs)
            return TensorModel(self.gtab, fit_method='RESTORE', sigma=sigma)

    def __check_gtab(self):
        "Check if the gradient table is available and create one if not"
        if self.gtab is None:
            print("No gradient table found. Creating one with default params.")
            self.gtab = self.make_gradient_table()

    def __median_otsu(self, nifti: nib.Nifti1Image, median_radius=2, numpass=1,
                     **kwargs) -> tuple[np.ndarray, np.ndarray]:
        "Run the median_otsu function with default parameters."
        return median_otsu(
            nifti.get_fdata(),
            median_radius=median_radius,
            numpass=numpass,
            **kwargs
            )
