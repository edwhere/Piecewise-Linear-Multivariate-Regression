"""
Threshold decomposition operations. This module has the following callable functions:

   univar_decomp(xval, thvec)
        Performs threshold decomposition of a number using a single vector of Q distinct thresholds

   multivar_decomp_uniform(xmat, thvec)
        Performs threshold decomposition of a data matrix (numpy ndarray) of size M x N using a single vector
        of Q distinct thresholds

   multivar_decomp(xmat, thvec)
        Performs threshold decomposition of a data matrix (numpy ndarray) of size M x N using a list
        of N vectors where each vector defines the list of thresholds for each of the N dimensions.

   Usage examples:

   import numpy as np
   import thdecomp as th

   xdat = np.arange(12).reshape(4, 3)
   tdat = [[1.0, 2.0], [], [-1, 1, 5]]
   tvec = [-1.0, 2.0, 5.0]
   rval = 6.2

   Xgen = th.multivar_decomp(xdat, tdat)
   Xuni = th.multivar_decomp_uniform(xdat, tvec)
   Xval = th.univar_decomp(rval, tvec)

"""
import sys
import numpy as np


def multivar_decomp_uniform(xmat, thvec):
    """
    Compute the threshold decomposotion from data expressed as a numpy array using a single set of Q thresholds.
    The same set of thresholds is used to partition each of the N dimensions of the space; thus forming a uniform grid.
    :param xmat: A data matrix of size M x N represented as a numpy array
    :param thvec: A vector of Q distinct threshold values
    :return: A data matrix with M rows and N(Q+1) columns, which corresponds to the application of
    threshold decomposition to each of the N features (variables).
    """
    nsegs = len(thvec) + 1  # number of segments in a decomposition
    nrows = xmat.shape[0]  # number of rows in the data matrix and in the decomposed data matrix
    ncols = nsegs * xmat.shape[1]  # number of columns in the decomposed data matrix

    result = np.zeros((nrows, ncols))  # placeholder for the decomposed data matrix

    # Create an iterator with access to row, col indices
    it = np.nditer(xmat, flags=['multi_index'])

    while not it.finished:
        row, col = it.multi_index
        tvec = univar_decomp(it[0], thvec)

        k = col * nsegs  # index to starting column where threshold vector should be placed

        result[row, k:k + nsegs] = tvec

        it.iternext()

    return result


def multivar_decomp(xmat, thlist):
    """
    Compute a multivariate threshold decomposition of a data matrix of size M x N using multiple threshold vectors.
    There is one threshold vector for each of the N dimensions.
    :param xmat: A data matrix of size M x N represented as a numpy array
    :param thlist: A list of N lists. Each of the N lists is a threshold vector. Each vector has Qi thresholds.
    :return: A data matrix with M rows and N' columns, where N' = (Q1 + 1) + (Q2 +1) + ... + (QN +1)
    """
    # Find the number of rows and columns in the data matrix. Also the number of threshold vectors
    xrows, xcols = xmat.shape
    numvecs = len(thlist)

    # Verify that the number of threshold vectors is the same as the number of variables (features)
    if not (numvecs == xcols):
        print "Error: number of threshold vectors does not match number of features (variables)"
        sys.exit(1)

    # Calculate the number of segments that each threshold vector generates; ie. Qi + 1. The add all segments.
    qvals = map(lambda vec: len(vec) + 1, thlist)
    q_total = reduce(lambda x, y: x+y, qvals)

    # Define a placeholder for the matrix that will contain the results
    result = np.zeros((xrows, q_total))

    # Create an iterator with access to row, col indices in the data matrix
    it = np.nditer(xmat, flags=['multi_index'])

    # Iterate over each element in the data matrix and obtain its threshold decomposition
    k_curr = 0
    while not it.finished:
        row, col = it.multi_index

        if col == 0:
            k_curr = 0
        else:
            k_curr += qvals[col - 1]

        if len(thlist[col]) > 0:
            # Find a threshold decomposition if there is at least one threshold in the vector
            decvec = univar_decomp(it[0], thlist[col])
        else:
            # If the threshold vector is empty there is no decomposition
            decvec = it[0]

        result[row, k_curr:k_curr + qvals[col]] = decvec

        it.iternext()

    return result


def univar_decomp(x, thvec):
    """
    Compute the threshold decomposition of a single numeric value using a vector of Q thresholds.
    :param x: A numeric value
    :param thvec: A vector of Q distinct thresholds
    :return: A vector of Q+1 elements, which is the threshold decomposition of value x
    """
    thvec.sort()

    thdecom = []

    val_interval = (0, x) if x >= 0 else (x, 0)

    for k in range(len(thvec) + 1):
        if k == 0:
            sel_interval = (-np.inf, thvec[k])

        elif k == len(thvec):
            sel_interval = (thvec[k - 1], np.inf)

        else:
            sel_interval = (thvec[k - 1], thvec[k])

        # Threshold decomposition can be implemented as a vector of signed intersection lengths
        interlen = __intersect_length(val_interval, sel_interval)
        signedlen = interlen if x >= 0.0 else -interlen

        thdecom.append(signedlen)

    return thdecom


def __intersect_length(inter1, inter2):
    """
    Computes the intersection length between two intervals. Each interval is a tuple of real-valued numbers
    :param inter1: A first interval
    :param inter2: A second interval
    :return: The length of the intersection between the two intervals
    """
    a, b = inter1
    p, q = inter2

    if p < a:
        if q < a:
            return 0
        else:
            if q > b:
                return b - a
            else:
                return q - a
    else:
        if p > b:
            return 0
        else:
            if q < b:
                return q - p
            else:
                return b - p

if __name__ == "__main__":

    xdat = np.arange(12).reshape(4, 3)

    tdat = [[1.0, 2.0], [], [-1, 1, 5]]

    res = multivar_decomp(xdat, tdat)

    print xdat
    print tdat
    print res
