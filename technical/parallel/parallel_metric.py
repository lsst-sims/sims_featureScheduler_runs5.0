import numpy as np
from itertools import repeat
import datetime

from multiprocessing import Pool, shared_memory, Manager
import rubin_sim.maf_proto as maf
import sqlite3
import pandas as pd

from rubin_sim.data import get_baseline


class SharedNumpyArray:
    # from https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html
    def __init__(self, array):
        # create the shared memory location of the same size of the array
        self._shared = shared_memory.SharedMemory(create=True, size=array.nbytes)
        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape
        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )
        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        np.copyto(res, array)

    def read(self):
        """ Read array without copy.
        """
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        """Copy arrray
        """
        return np.copy(self.read_array())

    def unlink(self):
        """Unlink when done with data
        """
        self._shared.close()
        self._shared.unlink()


def call_single_indx(shared_data, shared_metric, shared_slicer, indx):
    # Stupid print to see when it stalls
    result = shared_slicer(shared_data.read(),
                           shared_metric, 
                           indx=[indx], skip_setup=True)
    return result


def launch_jobs(shared_data, slicer, metric, num_jobs=6):

    with Manager() as manager:
        manager.shared_metric = metric
        manager.shared_slicer = slicer
        # make the args iterable
        args = zip(repeat(shared_data),
                   repeat(manager.shared_metric),
                   repeat(manager.shared_slicer),
                   range(len(slicer)))
        with Pool(processes=num_jobs) as pool:
            result = pool.starmap(call_single_indx, args)
    result = np.concatenate(result)
    return result


if __name__ == "__main__":

    nside = 16
    fast_metric = False

    # Read in observations
    # Read in some visits
    observations = get_baseline()
    con = sqlite3.connect(observations)
    df = pd.read_sql("select * from observations;", con)
    visits_array = df.to_records(index=False)

    shared_data = SharedNumpyArray(visits_array)
    slicer = maf.Slicer(nside=nside)
    slicer.setup_slicer(visits_array)
    # Use faster metric for testing
    if fast_metric:
        metric = maf.MeanMetric(col="airmass")
    else:
        metric = maf.SNNSNMetric()

    t1 = datetime.datetime.now()
    result = launch_jobs(shared_data, slicer, metric, num_jobs=12)
    shared_data.unlink()

    t2 = datetime.datetime.now()
    print("time to run in parallel", t2-t1)

    np.savez("metric_parallel.npz", result=result)

    # run classic style
    sl = maf.Slicer(nside=nside)
    metric = maf.SNNSNMetric()

    # result = sl(visits_array, metric)

    t3 = datetime.datetime.now()
    print("p2, time to run single core", t3-t2)

    np.savez("metric_single.npz", result=result)


    
