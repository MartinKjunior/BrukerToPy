"Module for processing multiple DTI datasets in parallel."

from dipy_dti_fit import DiPyDTI
from mpire import WorkerPool
from path_handler import DiPyPathHandler, init

def multiprocess_dti(path: str, dti_col: str, id_col: str = "Study ID",
                     pipeline: list = None, kwargs: dict = None,
                     n_jobs: int = 2) -> None:
    """Use multiple CPU cores to process multiple DTI datasets in parallel.
    Requires bruker_to_py and mpire to be installed.

    Parameters
    ----------
    path : str
        Path to the directory containing the Bruker data.
    dti_col : str
        The column in the animal overview sheet to check for the scan id.
    id_col : str, optional
        The column in the animal overview sheet to check for the exam id,
        by default "Study ID"
    pipeline : list, optional
        The processing steps to run, 
        by default ["motion_correction", "denoise", "fit_dti"]
    kwargs : dict, optional
        kwargs are passed into the individual processing steps. kwargs
        should be a dictionary with keys corresponding to the steps in the
        pipeline and values should be dictionaries of additional arguments
        to pass into each step, by default {}
    n_jobs : int, optional
        Number of CPU cores to use, by default 2

    Returns
    -------
    list[TensorFit]
        List of the tensor fits for each dataset.

    Raises
    ------
    ValueError
        If degibbs is in the pipeline and num_processes is not 1.

    Examples
    --------
    >>> from multiprocess_wrappers import multiprocess_DTI
    >>> path = str(Path.cwd().parent / 'MRI_data')
    >>> multiprocess_DTI(
        path,
        dti_col = 'dti',
        id_col = 'MR #',
        n_jobs = 3
    )
    """
    if WorkerPool is None:
        raise ImportError("mpire not found.")
    kwargs = kwargs or {}
    pipeline = pipeline or ["motion_correction", "denoise", "fit_dti"]
    if kwargs.get('degibbs', {}).get('num_processes', 1) != 1:
        raise ValueError("No additional processes allowed for degibbs.")
    data_object = init(path, msg=False)
    with WorkerPool(n_jobs) as pool:
        pool.imap_unordered(
            _multiprocess_dti_helper,
            [(data_object, exam_id, dti_col, id_col, pipeline, kwargs)
             for exam_id in data_object.avail_exam_ids]
            )

def _multiprocess_dti_helper(data_object, exam_id, dti_col, id_col, pipeline, kwargs):
    path_handler = DiPyPathHandler(data_object, exam_id)
    dpdti = DiPyDTI(
        path_handler.get_data_paths(
            dti_col=dti_col, id_col=id_col, return_metadata=True
            )
        )
    dpdti.run_pipeline(pipeline=pipeline, kwargs=kwargs)
