.. _Corrplots_hdf5:

mRMRe in Python with rpy2
===============
::

    Author : Philipp Hoffmann
    Date   : 2016

A very short introduction to rpy2 and the R feature selection library mRMRe.
  More documentation can be found at:  

  https://cran.r-project.org/web/packages/mRMRe/index.html  

  http://pandas.pydata.org/pandas-docs/stable/r_interface.html  

  http://rpy2.readthedocs.io/en/version_2.8.x/

----

Installing R libraries and rpy2
-----------------------------



To start using R in python do the following:
First install R and rpy2:

.. code:: python

    pacman -S R
    pip install --user rpy2

Now start R and install the needed library and its dependencies with:

.. code:: python

    install.packages('mRMRe')

When on a headless server you either need to enable x-forwarding or
provide a mirror manually.
If your install target or its dependencies can't be installed because
R is out of date, download an older source.tar.gz from CRAN and try:

.. code:: python

    install.packages(path_to_file, repos = NULL, type="source")

.. code:: python

    from rpy2.robjects.packages import importr
    base = importr('base')
    mr = importr('mRMRe')

Importr Enables you to access all functions in a R library via the standard
libraryName.functionName
Autocomplete should be working, though some functions can have names
that differ slightly from the ones listed in the R documentation and
docstrings are often unavailable.

You can also do:

.. code:: python

    python import rpy2.robjects as ro
    import rpy2.rinterface as ri

in order to directly create R objects, but creating an Rdataframe doesn't seem to be easy.


Converting data to R compatible Rdataframes
-----------------------------

.. code:: python

    import pandas as pd
    import numpy as np
    from rpy2.robjects import pandas2ri
    dataframe = pd.DataFrame(...) 

The label needs to be numerical. Rescaling doesn't seem to influence the
feature selection.

.. code:: python

    dataframe['label'] = np.float64(np.append(np.full(300000, 1.0), 
                                              np.full(300000, 0.0)))

Pandas dataframes need to be converted to Rdataframes first.
Rdataframes cannot contain uints, there are no uints in R. The mRMRe
package requires all dataframe entries to be of numeric type, which
maps to a float in Python, so preprocess accordingly. 

The following can be pretty memory intesive. Make
sure to have up to 5 times more free memory than your process
is currently using.
(in my case it took 200mb baseline memory + 1.1gb for the
dataframe(apart from the size increase from bools, etc. → float64
pretty much the same size as the hdf5 file on disk) + 1.6gb for the
Rdataframe + 2.0gb for the mrmrData.

.. code:: python

    pandas2ri.activate()
    rdataframe = pandas2ri.py2ri(dataframe)



Running mRMRe and getting results
-----------------------------


.. code:: python

    mrmrData = mr.mRMR_data(data = (rdataframe))


Setting solution\_count=1 results in a standard mrmr procedure.
target\_indices is the index of the label, indices start with 1 not with
0!

Setting a higher solution\_count results in multiple mrmr selections
being done, each on a bootstrapped sample fraction of 1/sol\_count.

.. code:: python
    
    solutionCount = 5
    selectionEnsemble = mr.mRMR_ensemble(data = mrmrData, target_indices = 125, 
                  feature_count = 50, solution_count = solutionCount )


The output of mr.mRMR_ensemble can't be used directly. It has to be converted first.


.. code:: python

    ensembleWeights = [mr.featureNames(selectionEnsemble)[i-1] 
        for i in list(mr.solutions(selectionEnsemble)[0])]
    splitWeights = np.array_split(ensembleWeights, solution_count)


This would be the last step if you only used solution\_count=1.
The features in the solution vector are ordered, so the Jaccard index can be calculated without repeatedly calculating selections.

For ensembles you can construct a feature ranking by summing the rank
of each feature for each solution.
This should lead to less variance in the selection of features.

.. code:: python

	def attribute_vote(splitWeights):
    """Calculates the summed ranks of each feature in an ensemble
    of mRMR selections."""
     	solutionCount = len(ensembleSelection)
    	solutionLength =  len(np.concatenate(ensembleSelection))
    	attributes = set(np.concatenate(ensembleSelection))
    	attributes = {attr : 0 for attr in attributes}
    	for solution in ensembleSelection:
        	for i,attr in enumerate(solution):
            	attributes[attr] += i
    	attributes = [(attr, attributes[attr]) for attr in attributes]        
    	attributes = sorted(attributes, key=lambda tup: tup[1] )
    	return attributes
	
.. code:: python

    test = attribute_vote(ensembleWeights, 5)
    print(test)


Notes
----
The memory footprint can be minimized by spawning a subprocess that converts the pandas dataframe and returns the Rdataframe to another subprocess and then terminates. This frees the 1.1gb used by the dataframe. The same is done for the Rdataframe→mrmrData conversion. This brings the maximum needed memory to 3.5 times the size of the pandas dataframe, down from 4.5 times its size.

