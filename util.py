# -*- coding: utf-8 -*-
"""
Markov Decision Process (MDP) Toolbox: ''util'' module
=======================================================

The ''util'' module provides functions to check that an MDP is validly described. There are also functions for working with MDPs while they are being solved.
"""

import numpy as _np

import mdptoolbox.error as _error

_MDPERR = {
    "mat_nonneg":
        "Transition probabilities must be non-negative.",
    "mat_square":
        "A transition probabilities matrix must be square, with dimensions S*S.",
    "mat_stoch":
        "Each row of a transition probabilites matrix must sum to one (1).",
    "obj_shape":
        "Object arrays for transition probabilites and rewards "
        "must have only 1 dimension: the number of actions A. Each element of "
        "the object array contains an SxS ndarray or matrix.",
    "obj_square" :
        "Each element of an object array for transition "
        "probabilities and rewards must contain an SxS ndarray or matrix; i.e. "
        "P[a].shape = (S, S) or R[a].shape = (S, S).",
    "P_type" :
        "The transition probabilities must be in a numpy array; "
        "i.e. type(P) is ndarray.",
    "P_shape" :
        "The transition probability array must have the shape "
        "(A, S, S)  with S : number of states greater than 0 and A : number of "
        "actions greater than 0. i.e. R.shape = (A, S, S)",
    "PR_incompat" :
        "Incompatibility between P and R dimensions.",
    "R_type" :
        "The rewards must be in a numpy array; i.e. type(R) is "
        "ndarray, or numpy matrix; i.e. type(R) is matrix.",
    "R_shape" :
        "The reward matrix R must be an array of shape (A, S, S) or "
        "(S, A) with S : number of states greater than 0 and A : number of "
        "actions greater than 0. i.e. R.shape = (S, A) or (A, S, S)."
}

def _checkDimensionsListLike(arrays):
    """
    Check that each array in a list of arrays has the same size.

    :param arrays:
    :return:
    """
    dim1 = len(arrays)
    dim2, dim3 = arrays[0].shape
    for aa in range(1, dim1):
        dim2_aa, dim3_aa = arrays[aa].shape
        if (dim2_aa != dim2) or (dim3_aa != dim3):
            raise _error.InvalidError(_MDPERR["obj_square"])
    return dim1, dim2, dim3

def _checkRewardsListLike(reward, n_actions, n_states):
    """
    Check that a list-like reward input is valid.

    :param reward:
    :param n_actions:
    :param n_states:
    :return:
    """
    try:
        lenR = len(reward)
        if lenR == n_actions:
            dim1, dim2, dim3 = _checkDimensionsListLike(reward)
        elif lenR == n_states:
            dim1 = n_actions
            dim2 = dim3 = lenR
        else:
            raise _error.InvalidError(_MDPERR["R_shape"])
    except AttributeError:
        raise _error.InvalidError(_MDPERR["R_shape"])
    return dim1, dim2, dim3

def isSquare(matrix):
    """
    Check that ''matrix'' is square

    :param matrix:
    :return:
    """
    try:
        try:
            dim1, dim2 = matrix.shape
        except AttributeError:
            dim1, dim2 = _np.array(matrix).shape
    except ValueError:
        return False
    if dim1 == dim2:
        return True
    return False

def isStochastic(matrix):
    """
    Check that ''matrix'' is row stochastic

    :param matrix:
    :return:
    """
    try:
        absdiff = (_np.abs(matrix.sum(axis=1)-_np.ones(matrix.shape[0])))
    except AttributeError:
        matrix = _np.array(matrix)
        absdiff = (_np.abs(matrix.sum(axis=1)-_np.ones(matrix.shape[0])))
    return (absdiff.max() <= 10*_np.spacing(_np.float64(1)))

def isNonNegative(matrix):
    """
    Check that ''matrix'' is row non-negative

    :param matrix:
    :return:
    """
    try:
        if (matrix >= 0).all():
            return True
    except (NotImplementedError, AttributeError, TypeError):
        try:
            if (matrix.data >= 0).all():
                return True
        except AttributeError:
            matrix = _np.array(matrix)
            if (matrix.data >= 0).all():
                return True
    return False

def checkSquareStochastic(matrix):
    """
    Check if ''matrix'' is a square and row-stochastic.
    To pass the check the following conditions must be met:
    * The matrix should be square, so the number of columns equals the number of rows.
    * The matrix should be row-stochastic so the rows should sum to one.
    * Each value in the matrix must be positive

    :param matrix:
    :return:
    """
    if not isSquare(matrix):
        raise _error.SquareError
    if not isStochastic(matrix):
        raise _error.StochasticError
    if not isNonNegative(matrix):
        raise _error.NonNegativeError

def check(P, R):
    """
    Check if ''P'' and ''R'' define a valid Markov Decision Process (MDP).
    :param P:
    :param R:
    :return:
    """
    # Checking P
    try:
        if P.ndim == 3:
            aP, sP0, sP1 = P.shape
        elif P.ndim == 1:
            aP, sP0, sP1 = _checkDimensionsListLike(P)
        else:
            raise _error.InvalidError(_MDPERR["P_shape"])
    except AttributeError:
        try:
            aP, sP0, sP1 = _checkDimensionsListLike(P)
        except AttributeError:
            raise _error.InvalidError(_MDPERR["P_shape"])
    msg = ""
    if aP <= 0:
        msg = "The number of actions in P must be greater than 0."
    elif sP0 <= 0:
        msg = "The number of states in P must be greater than 0."
    if msg:
        raise _error.InvalidError(msg)
    # Checking R
    try:
        ndimR = R.ndim
        if ndimR == 1:
            aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
        elif ndimR == 2:
            sR0, aR = R.shape
            sR1 = sR0
        elif ndimR == 3:
            aR, sR0, sR1 = R.shape
        else:
            raise _error.InvalidError(_MDPERR["R_shape"])
    except AttributeError:
        aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
    msg = ""
    if sR0 <= 0:
        msg = "The number of states in R must be greater than 0."
    elif aR <= 0:
        msg = "The number of actions in R must be greater than 0."
    elif sR0 != sR1:
        msg = "The matrix R must be square with respect to states."
    elif sP0 != sR0:
        msg = "The number of states must agree in P and R."
    elif aP != aR:
        msg = "The number of actions must agree in P and R."
    if msg:
        raise _error.InvalidError(msg)
    # Check that the P's are square, stochastic and non-negative
    for aa in range(aP):
        checkSquareStochastic(P[aa])

def getSpan(array):
    """
    Return the span of 'array'

    :param array:
    :return:
    """
    return array.max() - array.min()


