from abc import ABCMeta, abstractmethod

from catsim import irt
import numpy
from scipy.optimize import fmin
from scipy.optimize import differential_evolution


class Estimator:
    """Base class for proficiency estimators"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(Estimator, self).__init__()

    @abstractmethod
    def estimate(
        self, response_vector: list, administered_items: numpy.ndarray, current_theta: float
    ) -> float:
        """Uses a hill-climbing algorithm to find and return the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        """
        pass


class HillClimbingEstimator(Estimator):
    """Estimator that uses a hill-climbing algorithm to maximize the likelihood function

    :param precision: number of decimal points of precision
    :param verbose: verbosity level of the maximization method
    """

    def __init__(self, precision: int=6, verbose: bool=False):
        super(HillClimbingEstimator, self).__init__()
        self.__precision = precision
        self.__verbose = verbose
        self.__iters = 0

    @property
    def evaluations(self) -> int:
        """Count the number of function evaluations during the most recent estimation process

        :returns: number of function evaluations"""
        return self.__iters

    def estimate(
        self,
        response_vector: list,
        administered_items: numpy.ndarray,
        current_theta: float=None
    ) -> float:
        """Uses a hill-climbing algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: not used by this selector
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        :rtype: float
        """

        if set(response_vector) == 1:
            return float('inf')
        elif set(response_vector) == 0:
            return float('-inf')

        lbound = min(administered_items[:, 1])
        ubound = max(administered_items[:, 1])
        best_theta = -float('inf')
        max_ll = -float('inf')

        self.__iters = 0

        for i in range(10):
            intervals = numpy.linspace(lbound, ubound, 10)
            if self.__verbose:
                print('Bounds: ' + str(lbound) + ' ' + str(ubound))
                print('Interval size: ' + str(intervals[1] - intervals[0]))

            for ii in intervals:
                self.__iters += 1
                ll = irt.logLik(ii, response_vector, administered_items)
                if ll > max_ll:
                    max_ll = ll

                    if self.__verbose:
                        print('Iteration: {0}, Theta: {1}, LL: {2}'.format(self.__iters, ii, ll))

                    if abs(best_theta - ii) < float('1e-' + str(self.__precision)):
                        return ii

                    best_theta = ii

                else:
                    lbound = best_theta - (intervals[1] - intervals[0])
                    ubound = ii
                    break

        return best_theta


class BinarySearchEstimator(Estimator):
    """Estimator that uses a binary search approach in the log-likelihood function domain to maximize it

    :param precision: number of decimal points of precision
    :param verbose: verbosity level of the maximization method
    """

    def __init__(self, precision: int=6, verbose: bool=False):
        super(BinarySearchEstimator, self).__init__()
        self.__precision = precision
        self.__verbose = verbose
        self.__iters = 0

    @property
    def evaluations(self):
        """Count the number of function evaluations during the most recent estimation process

        :returns: number of function evaluations"""
        return self.__iters

    def estimate(
        self,
        response_vector: list,
        administered_items: numpy.ndarray,
        current_theta: float=None
    ) -> float:
        """Uses a binary search algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: not used by this selector
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        """

        if set(response_vector) == 1:
            return float('inf')
        elif set(response_vector) == 0:
            return float('-inf')

        lbound = min(administered_items[:, 1])
        ubound = max(administered_items[:, 1])
        best_theta = -float('inf')
        self.__iters = 0

        while True:
            self.__iters += 1
            if self.__verbose:
                print('Bounds: ' + str(lbound) + ' ' + str(ubound))
                print(
                    'Iteration: {0}, Theta: {1}, LL: {2}'.format(
                        self.__iters, best_theta, irt.logLik(
                            best_theta, response_vector, administered_items
                        )
                    )
                )

            if irt.logLik(ubound, response_vector, administered_items) > irt.logLik(
                lbound, response_vector, administered_items
            ):

                if abs(best_theta - ubound) < float('1e-' + str(self.__precision)):
                    return ubound

                best_theta = ubound
                lbound += (ubound - lbound) / 2
            else:

                if abs(best_theta - lbound) < float('1e-' + str(self.__precision)):
                    return lbound

                best_theta = lbound
                ubound -= (ubound - lbound) / 2


class FMinEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.fmin` to minimize the negative log-likelihood function"""

    def __init__(self):
        super(FMinEstimator, self).__init__()

    def estimate(
        self, response_vector: list, administered_items: numpy.ndarray, current_theta: float
    ) -> float:
        """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
        that minimizes the negative log-likelihood function, given a response vector, a
        matrix with the administered items parameters and the current :math:`\\theta`.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        """
        return fmin(irt.negativelogLik, current_theta, (response_vector, administered_items))


class DifferentialEvolutionEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.differential_evolution` to minimize the negative log-likelihood function

    :param bounds: a tuple containing both lower and upper bounds for the differential
                   evolution algorithm search space. In theory, it is best if they
                   represent the minimum and maximum possible :math:`\\theta` values;
                   in practice, one could also use the smallest and largest difficulty
                   parameters in the item bank, in case no better bounds for
                   :math:`\\theta` exist.
    """

    def __init__(self, bounds: tuple):
        super(DifferentialEvolutionEstimator, self).__init__()
        self.__lower_bound = min(bounds)
        self.__upper_bound = max(bounds)
        self.__iters = 0

    @property
    def evaluations(self):
        """Count the number of function evaluations during the most recent estimation process

        :returns: number of function evaluations"""
        return self.__iters

    def estimate(
        self,
        response_vector: list,
        administered_items: numpy.ndarray,
        current_theta: float=None
    ) -> float:
        """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
        that minimizes the negative log-likelihood function, given a response vector, a
        matrix with the administered items parameters and the current :math:`\\theta`.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :returns: not used by this selector
        """

        res = differential_evolution(
            irt.negativelogLik,
            bounds=[
                [self.__lower_bound * 2, self.__upper_bound * 2]
            ],
            args=(response_vector, administered_items)
        )

        self.__iters = res.nfev

        return res.x[0]
