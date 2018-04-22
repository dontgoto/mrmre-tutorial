from rpy2.robjects.packages import importr
import numpy as np
import h5py as h5
import pandas as pd
import glob

### a short testcase of MRMRe using IceCube simulation data
### still has a dependency on my preprocessing functions that is deactivated here
dontusePF = True
if dontusePF is False:
    from preprocessing_functions import preprocessing_functions as pf

base = importr('base')
mr = importr('mRMRe')
# path = '/home/fongo/sync/bachelorarbeit/daten/sim/'
path = '/fhgfs/users/phoffman/daten/sim/'
background = [h5.File(file, 'r') for file in sorted(glob.glob(path+'L3*12550*newnomissing'))]
signal = [h5.File(file, 'r') for file in sorted(glob.glob(path+'L3*14550*newnomissing'))]
colblacklist = ['Run', 'SubEventStream', 'SubEvent', 'Event', 'exists', 'pdg_encoding','type','shape', 'x', 'y', 'z', 'location', 'azimuth']

def generate_dataframe_from_hdf5(h5Array, signalLength, backgroundLength, blacklist=None, colblacklist=None):
    """Takes an array with h5Files as elements and appends them to create a pandas dataframe
    h5Array should be partitioned into signal and background, with signals coming first"""

    if (blacklist is None) and (dontusePF is False):
        blacklist=pf.blacklist

    dataframe = pd.DataFrame({key+'__'+col :
                              np.concatenate([h5File[key][col] for h5File in h5Array]).astype(float)
                              for key in [keys for keys in h5Array[0].keys()]
                              if key not in blacklist
                              if key not in ['honda2014_spl_solmin', 'I3MCPrimary']
                              for i, col in
                              enumerate(h5Array[0][key].dtype.names)
                              if col not in colblacklist
                             })
    return dataframe


def generate_mrmrData_from_pdDataframe(pdDataframe):
    """Generates an mrmrData object from a pandas Dataframe. This object can be used with the R MRMRe function."""
    mr = importr('mRMRe')
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    rDataframe = pandas2ri.py2ri(pdDataframe)
    mrmrData = mr.mRMR_data(data = (rDataframe))
    return mrmrData


def attribute_vote(ensembleSelection, solutionCount):
    """
    Parameters
    ----------
    ensembleSelection : np.array
        Array that contains the features that were selected by MRMRe in each cross validation run, no subarrays since MRMRe outputs a plain array. An ndim array is created by providing solutionCount
    solutionCount : int
        Number of validation runs done by MRMRe, used to split the array in an ndarray

    Returns
    -------
    sortedAttributes : np.array
        Contains the attributes selected in the final solution, sorted by majority vote from each sub solution
    """
    attributes = set(ensembleSelection)
    attributes = {attr : len(ensembleSelection) for attr in attributes}
    ensembleSelection = np.array(ensembleSelection)
    ensembleSelection = np.reshape(ensembleSelection, (solutionCount, -1))
    for solution in ensembleSelection:
        for i,attr in enumerate(solution):
            attributes[attr] += i
    sortedAttributes = {key : attributes[key] for key in sorted(attributes, key=attributes.get, reverse=True)}
    return sortedAttributes

### short example

# dataframe['label'] = np.float64(np.append(np.full(2083578, 1.0), np.full(2083578, 0.0)))
# selectionEnsemble = mr.mRMR_ensemble(data = mrmrData, target_indices = 125,
              # feature_count = 60, solution_count = 5)
# ensembleWeights = [mr.featureNames(selectionEnsemble)[i-1]
 # for i in list(mr.solutions(selectionEnsemble)[0])]

# sortedAttributes = attribute_vote(ensembleWeights, 5)
