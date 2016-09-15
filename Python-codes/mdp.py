# -*- coding: utf-8 -*-
"""
Markov Decision Process (MDP) Toolbox: ''mdp'' module
=======================================================

The ''mdp'' module provides classes for the resolution of descrete-time Markov Decision Processes.
"""

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

import mdptoolbox.util as _util

_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."


def _computeDimensions(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError:
        S = transition[0].shape[0]
    return S, A


def _printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print ("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print ("{:>10}{:>12d}".format(iteration, variation))
    else:
        print ("{:>10}{:>12}".format(iteration, variation))


if __name__ == '__main__':
    class MDP(object):
        """
        A Markove Decision Problem.
        Let ''S'' = the number of states, and ''A'' = the number of actions.

        """

        def ___init__(self, transitions, reward, discount, epsilon, max_iter, skip_check=False):
            # Initialise a MDP based on the input parameters.

            # if the discount is None then the algorithm is assumed to not use it
            # in its computations
            if discount is not None:
                self.discount = float(discount)
                assert 0.0 < self.discount <= 1.0, (
                    "Discount rate must be in [0; 1]"
                )
                if self.discount == 1:
                    print ("WARNING: check conditions of convergence. With no "
                           "discount, convergence can not be assumed.")

            # if the max_iter is None then the algorithm is assumed to not use it
            # in its computations
            if max_iter is not None:
                self.max_iter = int(max_iter)
                assert self.max_iter > 0, (
                    "The maximum number of iterations must be greater than 0."
                )

            # check that epsilon is something sane
            if epsilon is not None:
                self.epsilon = float(epsilon)
                assert self.epsilon > 0, "Epsilon must be greater than 0."

            if not skip_check:
                # We run a check on P and R to make sure they are describing an
                # MDP. If an exception isn't raised then they are assumed to be
                # correct.
                _util.check(transitions, reward)

            self.S, self.A = _computeDimensions(transitions)
            self.P = self._computeTransition(transitions)
            self.R = self._computeReward(reward, transitions)

            # the verbosity is by default turned off
            self.verbose = False
            # Initially the time taken to perform the computations is set to None
            self.time = None
            # set the initial iteration count to zero
            self.iter = 0
            # V should be stored as a vector ie shape of (S,) or (1, S)
            self.V = None
            # policy can also be stored as a vector
            self.policy = None

        def __repr__(self):
            P_repr = "P: \n"
            R_repr = "R: \n"
            for aa in range(self.A):
                P_repr += repr(self.P[aa]) + "\n"
                R_repr += repr(self.R[aa]) + "\n"
            return (P_repr + "\n" + R_repr)

        def _bellmanOperator(self, V=None):
            # Apply the Bellman operator on the value function.
            #
            # Updates the value function and the Vprev-improving policy.
            #
            # Returns: (policy, value), tuple of new policy and its value
            #
            # If V hasn't been sent into the method, then we assume to be working
            # on the objects V attribute
            if V is None:
                # this V should be a reference to the data rather than a copy
                V = self.V
            else:
                # make sure the user supplied V is of the right shape
                try:
                    assert V.shape in ((self.S,), (1, self.S)), "V is not the " \
                                                                "right shape (Bellman operator)."
                except AttributeError:
                    raise TypeError("V must be a numpy array or matrix.")
            # Looping through each action the Q-value matrix is calculated.
            # P and V can be any object that supports indexing, so it is important
            # that you know they define a valid MDP before calling the
            # _bellmanOperator method. Otherwise the results will be meaningless.
            Q = _np.empty((self.A, self.S))
            if __name__ == '__main__':
                for aa in range(self.A):
                    Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
                # Get the policy and value, for now it is being returned but ...
                # which way is better?
                # 1. Return, (policy, value)
                return (Q.argmax(axis=0), Q.max(axis=0))
                # 2.update self.policy and self.V directly
                # self.V = Q.max(axis=1)
                # self.policy = Q.argmax(axis=1)

        def _computeTransition(self, transition):
            return tuple(transition[a] for a in range(self.A))

        def _computeReward(self, reward, transition):
            # Compute the reward for the system in one state chosing an action.
            try:
                if reward.ndim == 1:
                    return self._computeVectorReward(reward)
                elif reward.ndim == 2:
                    return self._computeArrayReward(reward)
                else:
                    r = tuple(map(self._computeMatrixReward, reward, transition))
                    return r
            except (AttributeError, ValueError):
                if len(reward) == self.A:
                    r = tuple(map(self._computeMatrixReward, reward, transition))
                    return r
                else:
                    return self._computeVectorReward(reward)

        def _computeVectorReward(self, reward):
            if _sp.issparse(reward):
                raise NotImplementedError
            else:
                r = _np.array(reward).reshape(self.S)

                return tuple(r for a in range(self.A))

        def _computeArrayReward(self, reward):
            if _sp.issparse(reward):
                raise NotImplementedError
            else:
                def func(x):
                    return _np.array(x).reshape(self.S)

                return tuple(func(reward[:, a]) for a in range(self.A))

        def _computeMatrixReward(self, reward, transition):
            if _sp.issparse(reward):
                return reward.multiply(transition).sum(1).A.reshape(self.S)
            elif _sp.issparse(transition):
                return transition.multiply(reward).sum(1).A.reshape(self.S)
            else:
                return _np.mutltiply(transition, reward).sum(1).reshape(self.S)

        def _startRun(self):
            if self.verbose:
                _printVerbosity('Iteration', 'Variation')

            self.time = _time.time()

        def _endRun(self):
            # store value and policy as tuples
            self.V = tuple(self.V.tolist())

            try:
                self.policy = tuple(self.policy.tolist())
            except AttributeError:
                self.policy = tuple(self.policy)

            self.time = _time.time() - self.time

        def run(self):
            """
            Raises error because child classes should implement this function.
            :return:
            """
            raise NotImplementedError("You should create a run() method.")

        def setSilent(self):
            """Set the MDP algorithm to silent mode."""
            self.verbose = False

        def setVerbose(self):
            """Set the MDP algorithm to verbose mode."""
            self.verbose = True


    class FiniteHorizon(MDP):

        """A MDP solved using the finite-horizon backwards induction algorithm.
        """

        def __init__(self, transitions, reward, discount, N, h=None,
                     skip_check=False):
            # Initialise a finite horizon MDP.
            self.N = int(N)
            assert self.N > 0, "N must be greater than 0."
            # Initialise the base class
            MDP.__init__(self, transitions, reward, discount, None, None,
                         skip_check=skip_check)
            # remove the iteration counter, it is not meaningful for backwards
            # induction
            del self.iter
            # There are value vectors for each time step up to the horizon
            self.V = _np.zeros((self.S, N + 1))
            # There are policy vectors for each time step before the horizon, when
            # we reach the horizon we don't need to make decisions anymore.
            self.policy = _np.empty((self.S, N), dtype=int)
            # Set the reward for the final transition to h, if specified.
            if h is not None:
                self.V[:, N] = h

        def run(self):
            # Run the finite horizon algorithm.
            self.time = _time.time()
            # loop through each time period
            for n in range(self.N):
                W, X = self._bellmanOperator(self.V[:, self.N - n])
                stage = self.N - n - 1
                self.V[:, stage] = X
                self.policy[:, stage] = W
                if self.verbose:
                    print(("stage: %s, policy: %s") % (
                        stage, self.policy[:, stage].tolist()))
            # update time spent running
            self.time = _time.time() - self.time
            # After this we could create a tuple of tuples for the values and
            # policies.
            # self.V = tuple(tuple(self.V[:, n].tolist()) for n in range(self.N))
            # self.policy = tuple(tuple(self.policy[:, n].tolist())
            #                    for n in range(self.N))


    class _LP(MDP):
        """A discounted MDP soloved using linear programming.
        This class requires the Python ``cvxopt`` module to be installed.
        """
        def __init__(self, transitions, reward, discount, skip_check=False):
            # Initialise a linear programming MDP.
            # import some functions from cvxopt and set them as object methods
            try:
                from cvxopt import matrix, solvers
                self._linprog = solvers.lp
                self._cvxmat = matrix
            except ImportError:
                raise ImportError("The python module cvxopt is required to use "
                                  "linear programming functionality.")
            # initialise the MDP. epsilon and max_iter are not needed
            MDP.__init__(self, transitions, reward, discount, None, None,
                         skip_check=skip_check)
            # Set the cvxopt solver to be quiet by default, but ...
            # this doesn't do what I want it to do c.f. issue #3
            if not self.verbose:
                solvers.options['show_progress'] = False

        def run(self):
            # Run the linear programming algorithm.
            self.time = _time.time()
            f = self._cvxmat(_np.ones((self.S, 1)))
            h = _np.array(self.R).reshape(self.S * self.A, 1, order="F")
            h = self._cvxmat(h, tc='d')
            M = _np.zeros((self.A * self.S, self.S))
            for aa in range(self.A):
                pos = (aa + 1) * self.S
                M[(pos - self.S):pos, :] = (
                    self.discount * self.P[aa] - _sp.eye(self.S, self.S))
            M = self._cvxmat(M)
            self.V = _np.array(self._linprog(f, M, -h)['x']).reshape(self.S)
            # apply the Bellman operator
            self.policy, self.V = self._bellmanOperator()
            # update the time spent solving
            self.time = _time.time() - self.time
            # store value and policy as tuples
            self.V = tuple(self.V.tolist())
            self.policy = tuple(self.policy.tolist())


    class PolicyIteration(MDP):
        """A discounted MDP solved using the policy iteration algorithm.
        """
        def __init__(self, transitions, reward, discount, policy0=None,
                     max_iter=1000, eval_type=0, skip_check=False):
            # Initialise a policy iteration MDP.
            #
            # Set up the MDP, but don't need to worry about epsilon values
            MDP.__init__(self, transitions, reward, discount, None, max_iter,
                         skip_check=skip_check)
            # Check if the user has supplied an initial policy. If not make one.
            if policy0 is None:
                # Initialise the policy to the one which maximises the expected
                # immediate reward
                null = _np.zeros(self.S)
                self.policy, null = self._bellmanOperator(null)
                del null
            else:
                # Use the policy that the user supplied
                # Make sure it is a numpy array
                policy0 = _np.array(policy0)
                # Make sure the policy is the right size and shape
                assert policy0.shape in ((self.S,), (self.S, 1), (1, self.S)), \
                    "'policy0' must a vector with length S."
                # reshape the policy to be a vector
                policy0 = policy0.reshape(self.S)
                # The policy can only contain integers between 0 and S-1
                msg = "'policy0' must be a vector of integers between 0 and S-1."
                assert not _np.mod(policy0, 1).any(), msg
                assert (policy0 >= 0).all(), msg
                assert (policy0 < self.S).all(), msg
                self.policy = policy0
            # set the initial values to zero
            self.V = _np.zeros(self.S)
            # Do some setup depending on the evaluation type
            if eval_type in (0, "matrix"):
                self.eval_type = "matrix"
            elif eval_type in (1, "iterative"):
                self.eval_type = "iterative"
            else:
                raise ValueError("'eval_type' should be '0' for matrix evaluation "
                                 "or '1' for iterative evaluation. The strings "
                                 "'matrix' and 'iterative' can also be used.")

        def _computePpolicyPRpolicy(self):
            # Compute the transition matrix and the reward matrix for a policy.
            #
            Ppolicy = _np.empty((self.S, self.S))
            Rpolicy = _np.zeros(self.S)
            for aa in range(self.A):  # avoid looping over S
                # the rows that use action a.
                ind = (self.policy == aa).nonzero()[0]
                # if no rows use action a, then no need to assign this
                if ind.size > 0:
                    try:
                        Ppolicy[ind, :] = self.P[aa][ind, :]
                    except ValueError:
                        Ppolicy[ind, :] = self.P[aa][ind, :].todense()
                    # PR = self._computePR() # an apparently uneeded line, and
                    # perhaps harmful in this implementation c.f.
                    # mdp_computePpolicyPRpolicy.m
                    Rpolicy[ind] = self.R[aa][ind]
            # self.R cannot be sparse with the code in its current condition, but
            # it should be possible in the future. Also, if R is so big that its
            # a good idea to use a sparse matrix for it, then converting PRpolicy
            # from a dense to sparse matrix doesn't seem very memory efficient
            if type(self.R) is _sp.csr_matrix:
                Rpolicy = _sp.csr_matrix(Rpolicy)
            # self.Ppolicy = Ppolicy
            # self.Rpolicy = Rpolicy
            return (Ppolicy, Rpolicy)

        def _evalPolicyIterative(self, V0=0, epsilon=0.0001, max_iter=10000):
            # Evaluate a policy using iteration.
            #
            try:
                assert V0.shape in ((self.S,), (self.S, 1), (1, self.S)), \
                    "'V0' must be a vector of length S."
                policy_V = _np.array(V0).reshape(self.S)
            except AttributeError:
                if V0 == 0:
                    policy_V = _np.zeros(self.S)
                else:
                    policy_V = _np.array(V0).reshape(self.S)

            policy_P, policy_R = self._computePpolicyPRpolicy()

            if self.verbose:
                _printVerbosity("Iteration", "V variation")

            itr = 0
            done = False
            while not done:
                itr += 1

                Vprev = policy_V
                policy_V = policy_R + self.discount * policy_P.dot(Vprev)

                variation = _np.absolute(policy_V - Vprev).max()
                if self.verbose:
                    _printVerbosity(itr, variation)

                # ensure |Vn - Vpolicy| < epsilon
                if variation < ((1 - self.discount) / self.discount) * epsilon:
                    done = True
                    if self.verbose:
                        print(_MSG_STOP_EPSILON_OPTIMAL_VALUE)
                elif itr == max_iter:
                    done = True
                    if self.verbose:
                        print(_MSG_STOP_MAX_ITER)

            self.V = policy_V

        def _evalPolicyMatrix(self):
            # Evaluate the value function of the policy using linear equations.
            #
            Ppolicy, Rpolicy = self._computePpolicyPRpolicy()
            # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
            self.V = _np.linalg.solve(
                (_sp.eye(self.S, self.S) - self.discount * Ppolicy), Rpolicy)

        def run(self):
            # Run the policy iteration algorithm.
            self._startRun()

            while True:
                self.iter += 1
                # these _evalPolicy* functions will update the classes value
                # attribute
                if self.eval_type == "matrix":
                    self._evalPolicyMatrix()
                elif self.eval_type == "iterative":
                    self._evalPolicyIterative()
                # This should update the classes policy attribute but leave the
                # value alone
                policy_next, null = self._bellmanOperator()
                del null
                # calculate in how many places does the old policy disagree with
                # the new policy
                n_different = (policy_next != self.policy).sum()
                # if verbose then continue printing a table
                if self.verbose:
                    _printVerbosity(self.iter, n_different)
                # Once the policy is unchanging of the maximum number of
                # of iterations has been reached then stop
                if n_different == 0:
                    if self.verbose:
                        print(_MSG_STOP_UNCHANGING_POLICY)
                    break
                elif self.iter == self.max_iter:
                    if self.verbose:
                        print(_MSG_STOP_MAX_ITER)
                    break
                else:
                    self.policy = policy_next

            self._endRun()


    class PolicyIterationModified(PolicyIteration):
        """A discounted MDP  solved using a modifified policy iteration algorithm.
        Arguments
        """
        def __init__(self, transitions, reward, discount, epsilon=0.01,
                     max_iter=10, skip_check=False):
            # Initialise a (modified) policy iteration MDP.

            # Maybe its better not to subclass from PolicyIteration, because the
            # initialisation of the two are quite different. eg there is policy0
            # being calculated here which doesn't need to be. The only thing that
            # is needed from the PolicyIteration class is the _evalPolicyIterative
            # function. Perhaps there is a better way to do it?
            PolicyIteration.__init__(self, transitions, reward, discount, None,
                                     max_iter, 1, skip_check=skip_check)

            # PolicyIteration doesn't pass epsilon to MDP.__init__() so we will
            # check it here
            self.epsilon = float(epsilon)
            assert epsilon > 0, "'epsilon' must be greater than 0."

            # computation of threshold of variation for V for an epsilon-optimal
            # policy
            if self.discount != 1:
                self.thresh = self.epsilon * (1 - self.discount) / self.discount
            else:
                self.thresh = self.epsilon

            if self.discount == 1:
                self.V = _np.zeros(self.S)
            else:
                Rmin = min(R.min() for R in self.R)
                self.V = 1 / (1 - self.discount) * Rmin * _np.ones((self.S,))

        def run(self):
            # Run the modified policy iteration algorithm.

            self._startRun()

            while True:
                self.iter += 1

                self.policy, Vnext = self._bellmanOperator()
                # [Ppolicy, PRpolicy] = mdp_computePpolicyPRpolicy(P, PR, policy);

                variation = _util.getSpan(Vnext - self.V)
                if self.verbose:
                    _printVerbosity(self.iter, variation)

                self.V = Vnext
                if variation < self.thresh:
                    break
                else:
                    is_verbose = False
                    if self.verbose:
                        self.setSilent()
                        is_verbose = True

                    self._evalPolicyIterative(self.V, self.epsilon, self.max_iter)

                    if is_verbose:
                        self.setVerbose()

            self._endRun()


    class QLearning(MDP):
        """A discounted MDP solved using the Q learning algorithm.
        """
        def __init__(self, transitions, reward, discount, n_iter=10000,
                     skip_check=False):
            # Initialise a Q-learning MDP.

            # The following check won't be done in MDP()'s initialisation, so let's
            # do it here
            self.max_iter = int(n_iter)
            assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

            if not skip_check:
                # We don't want to send this to MDP because _computePR should not
                #  be run on it, so check that it defines an MDP
                _util.check(transitions, reward)

            # Store P, S, and A
            self.S, self.A = _computeDimensions(transitions)
            self.P = self._computeTransition(transitions)

            self.R = reward

            self.discount = discount

            # Initialisations
            self.Q = _np.zeros((self.S, self.A))
            self.mean_discrepancy = []

        def run(self):
            # Run the Q-learning algoritm.
            discrepancy = []

            self.time = _time.time()

            # initial state choice
            s = _np.random.randint(0, self.S)

            for n in range(1, self.max_iter + 1):

                # Reinitialisation of trajectories every 100 transitions
                if (n % 100) == 0:
                    s = _np.random.randint(0, self.S)

                # Action choice : greedy with increasing probability
                # probability 1-(1/log(n+2)) can be changed
                pn = _np.random.random()
                if pn < (1 - (1 / _math.log(n + 2))):
                    # optimal_action = self.Q[s, :].max()
                    a = self.Q[s, :].argmax()
                else:
                    a = _np.random.randint(0, self.A)

                # Simulating next state s_new and reward associated to <s,s_new,a>
                p_s_new = _np.random.random()
                p = 0
                s_new = -1
                while (p < p_s_new) and (s_new < (self.S - 1)):
                    s_new = s_new + 1
                    p = p + self.P[a][s, s_new]

                try:
                    r = self.R[a][s, s_new]
                except IndexError:
                    try:
                        r = self.R[s, a]
                    except IndexError:
                        r = self.R[s]

                # Updating the value of Q
                # Decaying update coefficient (1/sqrt(n+2)) can be changed
                delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
                dQ = (1 / _math.sqrt(n + 2)) * delta
                self.Q[s, a] = self.Q[s, a] + dQ

                # current state is updated
                s = s_new

                # Computing and saving maximal values of the Q variation
                discrepancy.append(_np.absolute(dQ))

                # Computing means all over maximal Q variations values
                if len(discrepancy) == 100:
                    self.mean_discrepancy.append(_np.mean(discrepancy))
                    discrepancy = []

                # compute the value function and the policy
                self.V = self.Q.max(axis=1)
                self.policy = self.Q.argmax(axis=1)

            self._endRun()


    class RelativeValueIteration(MDP):
        """A MDP solved using the relative value iteration algorithm.
        Arguments
        """
        def __init__(self, transitions, reward, epsilon=0.01, max_iter=1000,
                     skip_check=False):
            # Initialise a relative value iteration MDP.

            MDP.__init__(self, transitions, reward, None, epsilon, max_iter,
                         skip_check=skip_check)

            self.epsilon = epsilon
            self.discount = 1

            self.V = _np.zeros(self.S)
            self.gain = 0  # self.U[self.S]

            self.average_reward = None

        def run(self):
            # Run the relative value iteration algorithm.

            self._startRun()

            while True:

                self.iter += 1

                self.policy, Vnext = self._bellmanOperator()
                Vnext = Vnext - self.gain

                variation = _util.getSpan(Vnext - self.V)

                if self.verbose:
                    _printVerbosity(self.iter, variation)

                if variation < self.epsilon:
                    self.average_reward = self.gain + (Vnext - self.V).min()
                    if self.verbose:
                        print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                    break
                elif self.iter == self.max_iter:
                    self.average_reward = self.gain + (Vnext - self.V).min()
                    if self.verbose:
                        print(_MSG_STOP_MAX_ITER)
                    break

                self.V = Vnext
                self.gain = float(self.V[self.S - 1])

            self._endRun()


    class ValueIteration(MDP):
        """A discounted MDP solved using the value iteration algorithm.
        """
        def __init__(self, transitions, reward, discount, epsilon=0.01,
                     max_iter=1000, initial_value=0, skip_check=False):
            # Initialise a value iteration MDP.

            MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                         skip_check=skip_check)

            # initialization of optional arguments
            if initial_value == 0:
                self.V = _np.zeros(self.S)
            else:
                assert len(initial_value) == self.S, "The initial value must be " \
                                                     "a vector of length S."
                self.V = _np.array(initial_value).reshape(self.S)
            if self.discount < 1:
                # compute a bound for the number of iterations and update the
                # stored value of self.max_iter
                self._boundIter(epsilon)
                # computation of threshold of variation for V for an epsilon-
                # optimal policy
                self.thresh = epsilon * (1 - self.discount) / self.discount
            else:  # discount == 1
                # threshold of variation for V for an epsilon-optimal policy
                self.thresh = epsilon

        def _boundIter(self, epsilon):
            # Compute a bound for the number of iterations.
            #
            # for the value iteration
            # algorithm to find an epsilon-optimal policy with use of span for the
            # stopping criterion
            #
            k = 0
            h = _np.zeros(self.S)

            for ss in range(self.S):
                PP = _np.zeros((self.A, self.S))
                for aa in range(self.A):
                    try:
                        PP[aa] = self.P[aa][:, ss]
                    except ValueError:
                        PP[aa] = self.P[aa][:, ss].todense().A1
                # minimum of the entire array.
                h[ss] = PP.min()

            k = 1 - h.sum()
            Vprev = self.V
            null, value = self._bellmanOperator()
            # p 201, Proposition 6.6.5
            span = _util.getSpan(value - Vprev)
            max_iter = (_math.log((epsilon * (1 - self.discount) / self.discount) /
                                  span) / _math.log(self.discount * k))
            # self.V = Vprev

            self.max_iter = int(_math.ceil(max_iter))

        def run(self):
            # Run the value iteration algorithm.
            self._startRun()

            while True:
                self.iter += 1

                Vprev = self.V.copy()

                # Bellman Operator: compute policy and value functions
                self.policy, self.V = self._bellmanOperator()

                # The values, based on Q. For the function "max()": the option
                # "axis" means the axis along which to operate. In this case it
                # finds the maximum of the the rows. (Operates along the columns?)
                variation = _util.getSpan(self.V - Vprev)

                if self.verbose:
                    _printVerbosity(self.iter, variation)

                if variation < self.thresh:
                    if self.verbose:
                        print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                    break
                elif self.iter == self.max_iter:
                    if self.verbose:
                        print(_MSG_STOP_MAX_ITER)
                    break

            self._endRun()


    class ValueIterationGS(ValueIteration):
        """
        A discounted MDP solved using the value iteration Gauss-Seidel algorithm.
        """
        def __init__(self, transitions, reward, discount, epsilon=0.01,
                     max_iter=10, initial_value=0, skip_check=False):
            # Initialise a value iteration Gauss-Seidel MDP.

            MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                         skip_check=skip_check)

            # initialization of optional arguments
            if initial_value == 0:
                self.V = _np.zeros(self.S)
            else:
                if len(initial_value) != self.S:
                    raise ValueError("The initial value must be a vector of "
                                     "length S.")
                else:
                    try:
                        self.V = initial_value.reshape(self.S)
                    except AttributeError:
                        self.V = _np.array(initial_value)
                    except:
                        raise
            if self.discount < 1:
                # compute a bound for the number of iterations and update the
                # stored value of self.max_iter
                self._boundIter(epsilon)
                # computation of threshold of variation for V for an epsilon-
                # optimal policy
                self.thresh = epsilon * (1 - self.discount) / self.discount
            else:  # discount == 1
                # threshold of variation for V for an epsilon-optimal policy
                self.thresh = epsilon

        def run(self):
            # Run the value iteration Gauss-Seidel algorithm.

            self._startRun()

            while True:
                self.iter += 1

                Vprev = self.V.copy()

                for s in range(self.S):
                    Q = [float(self.R[a][s] +
                               self.discount * self.P[a][s, :].dot(self.V))
                         for a in range(self.A)]

                    self.V[s] = max(Q)

                variation = _util.getSpan(self.V - Vprev)

                if self.verbose:
                    _printVerbosity(self.iter, variation)

                if variation < self.thresh:
                    if self.verbose:
                        print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                    break
                elif self.iter == self.max_iter:
                    if self.verbose:
                        print(_MSG_STOP_MAX_ITER)
                    break

            self.policy = []
            for s in range(self.S):
                Q = _np.zeros(self.A)
                for a in range(self.A):
                    Q[a] = (self.R[a][s] +
                            self.discount * self.P[a][s, :].dot(self.V))

                self.V[s] = Q.max()
                self.policy.append(int(Q.argmax()))

            self._endRun()
