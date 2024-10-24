"Module for handling paths for DiPy processing."

from pathlib import Path
from typing import Any

import bruker_to_py as btp

init = btp.init

class DiPyPathHandler:
    """
    Class to handle paths for DiPy processing.
    Definitions of ID numbers (following brkraw naming convention):
        - Exam ID: The MR number, number of a new exam card in paravision, 
        e.g. 230215.
        - Scan ID: The number of the scan, e.g. 10, shown in paravision as E10 
        on the exam card.
        - Reco ID: The reconstruction number, e.g. 1. The default image is 1, 
        any additional processing, such as returning phase images, will have
        a different reco id. Diffusion images are usually reco id 1.

    Parameters
    ----------
    data_object : btp.DataObject
        The DataObject instance from bruker_to_py.py.
    exam_id : int|str|list
        The exam id (MR number), e.g. 230215. list is only valid for 
        "dti_analysis" option.
    reco_id : int, optional
        The reco id of the diffusion scans, e.g. 1.
    msg : bool, optional
        Whether to show the folder structure of the data, by default False.

    Attributes
    ----------
    D : btp.DataObject|str
        The DataObject instance from bruker_to_py.py or path to data.
    exam_id : int
        The exam id (or study id), e.g. 230215.
    reco_id : int
        The reco id of the diffusion scans, e.g. 1.

    Methods
    -------
    find_scan_id(data_col: str, id_col: str = "Study ID", exam_id = None) -> int
        Find the scan id for a given exam id based on a column in the rat 
        overview sheet.
    get_data_paths(scan_id: int = None, id_col: str = "Scan ID", 
                   return_metadata = False) -> dict
        Retrieve the paths to the data, bvals, bvecs, and mask for a given scan 
        id.
    """
    def __init__(self, data_object: Any|str, exam_id: int|str|list,
                 reco_id: int = 1, msg: bool = False,
                 animal_overview: str = "animal_overview.xlsx") -> None:
        if isinstance(data_object, str):
            if not Path(data_object).exists():
                raise FileNotFoundError(f'{data_object} does not exist.')
            data_object = btp.init(
                data_object, msg=msg, animal_overview=animal_overview
                )
        self.data_object = data_object
        self.exam_id = self.__set_exam_id(exam_id)
        self.scan_id = None
        self.reco_id = reco_id

    def find_scan_id(self, data_col: str, id_col: str = "Study ID",
                     exam_id = None) -> int:
        '''
        Find the scan id for a given exam id based on a column in the 
        animal_overview sheet.

        Parameters
        ----------
        data_col : str
            The column to search for the scan id.
        id_col : str, optional
            The column to search for the exam id. The default is "Study ID".
        exam_id : int, optional
            The exam id. The default is None.

        Returns
        -------
        scan_id : int
            The scan id.

        '''
        if not exam_id:
            if self.exam_id is None:
                raise ValueError('No exam id set.')
            exam_id = self.exam_id
        if self.data_object.animal_overview is None:
            raise ValueError('No animal overview sheet found.')
        return self.data_object.animal_overview.loc[
            self.data_object.animal_overview[id_col] == exam_id, data_col
            ].astype(int).values[0]

    def get_data_paths(self, which: str = 'dti_fit', **kwargs) -> dict:
        options = ['dti_fit', 'velocity_map', 'dti_analysis']
        if which != 'dti_analysis' and isinstance(self.exam_id, list):
            raise ValueError('Multiple exam ids only valid for "dti_analysis" '
                             'option.')
        if which == options[0]:
            return self.__get_dti_fit(**kwargs)
        elif which == options[1]:
            return self.__get_velocity_map(**kwargs)
        elif which == options[2]:
            return self.__get_dti_analysis(**kwargs)
        else:
            raise ValueError('Please choose one of the following options: '
                             f'{options}.')

    def __get_dti_fit(self, dti_col: str = None, scan_id: int = None,
                       id_col: str = "Study ID", return_metadata = False
                       ) -> dict:
        """For dipy_dti_fit.py

        Retrieve the paths to the data, bvals, bvecs, and mask for a given 
        scan id. Mask is assumed to exist within a folder called BrainMask 
        within the savedir directory (e.g. 
        "../Processed_MRI_data/20230505_094718_230215_1_1/BrainMask}).

        Parameters
        ----------
        dti_col : str
            The column in self.data_object.animal_overview to check for the scan id.
        scan_id : int, optional
            The scan id. If None, the scan id will be found based on the exam 
            id, by default None
        id_col : str, optional
            The column in self.data_object.animal_overview to check for the exam id, 
            by default "Study ID"
        return_metadata : bool, optional
            Whether to add the methods, acqp, and visu_pars to the output 
            dictionary, these are OrderedDicts, not paths, by default False

        Returns
        -------
        dict[str, str]
            A dictionary with the paths to the data, bvals, bvecs, and mask.
        """
        if dti_col is None and scan_id is None:
            raise ValueError('Please provide either dti_col or scan_id.')
        if scan_id is None:
            scan_id = self.find_scan_id(dti_col, id_col)
        data_paths = {}
        print(f'Looking for data for {self.exam_id}_{scan_id}...')
        raw_data = self.data_object.pull_exam_data(
            self.exam_id, scan_id, self.reco_id, only_path=True
        )
        if return_metadata:
            data_paths['method'] = raw_data['method']
            data_paths['acqp'] = raw_data['acqp']
            data_paths['visu_pars'] = raw_data['recos']['visu_pars']
        data_paths['data_path'] = raw_data['recos']['path']
        data_paths['bvals_path'] = self.data_object.pull_processed_data(
            self.exam_id, 'BV', substring = f'{self.exam_id}_{scan_id}_bval',
            only_path=True
        )[0]
        data_paths['bvecs_path'] = self.data_object.pull_processed_data(
            self.exam_id, 'BV', substring = f'{self.exam_id}_{scan_id}_bvec',
            only_path=True
        )[0]
        data_paths['savedir'] = self.data_object.get_savedir(self.exam_id)
        try:
            data_paths['mask_path'] = self.data_object.pull_processed_data(
                self.exam_id, 'BM', substring = f'{self.exam_id}*mask',
                only_path=True
            )[0]
        except FileNotFoundError:
            data_paths['mask_path'] = ""
            print(f'No mask found for {self.exam_id}.')
        return data_paths

    def __get_velocity_map(self, dti_col: str = None, scan_id: int = None,
        id_col: str = "Study ID", mag_reco_id: int = 1, 
        phase_reco_id: int = 2
        ) -> dict:
        "For dipy_process_phase.py"
        if dti_col is None and scan_id is None:
            raise ValueError('Please provide either dti_col or scan_id.')
        if scan_id is None:
            scan_id = self.find_scan_id(dti_col, id_col)
        print(f'reco_ids are {mag_reco_id=} and {phase_reco_id=}.')
        data_paths = {}
        print(f'Looking for data for {self.exam_id}_{scan_id}...')
        mag_data = self.data_object.pull_exam_data(
            self.exam_id, scan_id, mag_reco_id, only_path=True
        )
        phase_data = self.data_object.pull_exam_data(
            self.exam_id, scan_id, phase_reco_id, only_path=True
        )
        data_paths['magnitude_path'] = mag_data['recos']['path']
        data_paths['phase_path'] = phase_data['recos']['path']
        data_paths['bvals_path'] = self.data_object.pull_processed_data(
            self.exam_id, 'BV', substring = f'{self.exam_id}_{scan_id}_bval',
            only_path=True
        )[0]
        data_paths['bvecs_path'] = self.data_object.pull_processed_data(
            self.exam_id, 'BV', substring = f'{self.exam_id}_{scan_id}_bvec',
            only_path=True
        )[0]
        data_paths['methods_path'] = mag_data['method']
        data_paths['savedir'] = self.data_object.get_savedir(self.exam_id)
        try:
            data_paths['mask_path'] = self.data_object.pull_processed_data(
                self.exam_id, 'BM', substring = f'{self.exam_id}*mask',
                only_path=True
            )[0]
        except FileNotFoundError:
            data_paths['mask_path'] = ""
            print(f'No mask found for {self.exam_id}.')
        return data_paths

    def __get_dti_analysis(self, dti_col: str, scan_id: int = None, 
                           id_col: str = 'Study ID') -> list[dict]:
        "For dipy_dti_analysis.py"
        if not isinstance(self.exam_id, list):
            raise ValueError('dti_analysis requires a list of exam ids.')
        data_paths = []
        for exam_id in self.exam_id:
            scan_id = self.find_scan_id(dti_col, id_col, exam_id)
            dwi = self.data_object.pull_processed_data(
                exam_id,
                ['DiPyDTI', '*', 'MotionCorrected'],
                to_dict = False,
                as_numpy = False,
                substring = f'{scan_id}_motion_corrected',
                only_path = True
            )[-1]
            mask = self.data_object.pull_processed_data(
                exam_id,
                'BM',
                to_dict = False,
                as_numpy = True,
                substring = 'brain_mask',
                only_path = True
            )[-1]
            md = self.data_object.pull_processed_data(
                exam_id,
                ['DiPyDTI', '*', 'DTIFit'],
                to_dict = False,
                as_numpy = False,
                substring = f'{scan_id}_mean_diffusivity',
                only_path = True
            )[-1]
            fa = self.data_object.pull_processed_data(
                exam_id,
                ['DiPyDTI', '*', 'DTIFit'],
                to_dict = False,
                as_numpy = False,
                substring = f'{scan_id}_fractional_anisotropy',
                only_path = True
            )[-1]
            savedir = next(Path(self.data_object.path).parent.glob('Processed_*'))
            method = self.data_object.pull_exam_data(
                exam_id, scan_id, self.reco_id, only_path = True
            )['method']
            data_paths.append({
                'exam_id': exam_id,
                'scan_id': scan_id,
                'dwi': dwi,
                'mask': mask,
                'md': md,
                'fa': fa,
                'savedir': savedir,
                'data_label': dti_col,
                'method': method
            })
        return data_paths

    def __set_exam_id(self, exam_id: int|str|list):
        try:
            if isinstance(exam_id, list):
                return [int(i) for i in exam_id]
            return int(exam_id)
        except ValueError as e:
            print('exam_id must be an integer or a list of integers.')
            print(f'Error: {e}')
