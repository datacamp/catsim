import numpy
from scipy.optimize import differential_evolution
from scipy.stats import norm

from catsim import irt, cat
from catsim.simulation import Estimator


class HillClimbingEstimator(Estimator):
    """Estimator that uses a hill-climbing algorithm to maximize the likelihood function

    :param precision: number of decimal points of precision
    :param verbose: verbosity level of the maximization method
    """

    def __str__(self):
        return 'Hill Climbing Estimator'

    def __init__(self,
            precision: int = 6,
            dodd: bool = False,
            bounds: tuple = (-6, 6),
            verbose: bool = False
            ):
        super().__init__()
        self._precision = precision
        self._verbose = verbose
        self._evaluations = 0
        self._calls = 0
        self._dodd = dodd
        self._bounds = bounds

    @property
    def calls(self) -> float:
        """How many times the estimator has been called to maximize/minimize the log-likelihood function

        :returns: number of times the estimator has been called to maximize/minimize the log-likelihood function"""
        return self._calls

    @property
    def evaluations(self) -> float:
        """Total number of times the estimator has evaluated the log-likelihood function during its existence

        :returns: number of function evaluations"""
        return self._evaluations

    @property
    def avg_evaluations(self) -> float:
        """Average number of function evaluations for all tests the estimator has been used

        :returns: average number of function evaluations"""
        return self._evaluations / self._calls

    @property
    def dodd(self) -> bool:
        """Whether Dodd's method will be called by estimator in case the response vector
        is composed solely of right or wrong answers.

        :returns: boolean value indicating if Dodd's method will be used or not."""
        return self._dodd

    def _bound_estimate(self, estimate):
        lbound, ubound = self._bounds
        return min(max(estimate, lbound), ubound)

    def estimate(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        response_vector: list = None,
        est_theta: float = None,
        **kwargs
    ) -> float:
        """Returns the theta value that minimizes the negative log-likelihood function, given the current state of the
         test for the given examinee.

        :param index: index of the current examinee in the simulator
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param response_vector: a boolean list containing the examinee's answers to the administered items
        :param est_theta: a float containing the current estimated proficiency
        :returns: the current :math:`\\hat\\theta`
        """
        if (index is None or self.simulator is None) and (
            items is None and administered_items is None or response_vector is None or
            est_theta is None
        ):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and response_vector is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            response_vector = self.simulator.response_vectors[index]
            est_theta = self.simulator.latest_estimations[index]

        self._calls += 1

        # need to constrain all estimates between these bounds, rather then, e.g.
        # min / max difficulties
        lower_bound, upper_bound = self._bounds

        if len(set(response_vector)) == 1 and self._dodd:
            # append bounds in mock "items", so that the dodd procedure will
            # at least step toward the bounds we set. Note that this is stretching
            # the use of the term dodd.
            min_item = [0, lower_bound, 0, 0]
            max_item = [0, upper_bound, 0, 0]
            bound_items = numpy.vstack([min_item, max_item])
            return cat.dodd(est_theta, bound_items, response_vector[-1])

        if set(response_vector) == 1:
            return float('inf')
        elif set(response_vector) == 0:
            return float('-inf')

        best_theta = float('-inf')
        max_ll = float('-inf')

        self._evaluations = 0

        for _ in range(10):
            intervals = numpy.linspace(lower_bound, upper_bound, 10)
            if self._verbose:
                print(('Bounds: ' + str(lower_bound) + ' ' + str(upper_bound)))
                print(('Interval size: ' + str(intervals[1] - intervals[0])))

            for ii in intervals:
                self._evaluations += 1
                ll = irt.log_likelihood(ii, response_vector, items[administered_items])
                if ll > max_ll:
                    max_ll = ll

                    if self._verbose:
                        print(
                            (
                                'Iteration: {0}, Theta: {1}, LL: {2}'.format(
                                    self._evaluations, ii, ll
                                )
                            )
                        )

                    if abs(best_theta - ii) < float('1e-' + str(self._precision)):
                        return self._bound_estimate(ii)

                    best_theta = ii

                else:
                    lower_bound = best_theta - (intervals[1] - intervals[0])
                    upper_bound = ii
                    # reset best_theta, in case optimum is to the left of it
                    max_ll = float('-inf')
                    break

        return self._bound_estimate(best_theta)


class DifferentialEvolutionEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.differential_evolution` to minimize the negative log-likelihood function

    :param bounds: a tuple containing both lower and upper bounds for the differential
                   evolution algorithm search space. In theory, it is best if they
                   represent the minimum and maximum possible :math:`\\theta` values;
                   in practice, one could also use the smallest and largest difficulty
                   parameters in the item bank, in case no better bounds for
                   :math:`\\theta` exist.
    """

    def __str__(self):
        return 'Differential Evolution Estimator'

    def __init__(self, bounds: tuple):
        super(DifferentialEvolutionEstimator, self).__init__()
        self._lower_bound = min(bounds)
        self._upper_bound = max(bounds)
        self._evaluations = 0
        self._calls = 0

    @property
    def calls(self):
        """How many times the estimator has been called to maximize/minimize the log-likelihood function

        :returns: number of times the estimator has been called to maximize/minimize the log-likelihood function"""
        return self._calls

    @property
    def evaluations(self):
        """Total number of times the estimator has evaluated the log-likelihood function during its existence

        :returns: number of function evaluations"""
        return self._evaluations

    @property
    def avg_evaluations(self):
        """Average number of function evaluations for all tests the estimator has been used

        :returns: average number of function evaluations"""
        return self._evaluations / self._calls

    def estimate(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        response_vector: list = None,
        **kwargs
    ) -> float:
        """Uses :py:func:`scipy.optimize.differential_evolution` to return the theta value
        that minimizes the negative log-likelihood function, given the current state of the
        test for the given examinee.

        :param index: index of the current examinee in the simulator
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param response_vector: a boolean list containing the examinee's answers to the administered items
        :returns: the current :math:`\\hat\\theta`
        """
        if (index is None or self.simulator is None
            ) and (items is None and administered_items is None or response_vector is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and response_vector is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            response_vector = self.simulator.response_vectors[index]

        self._calls += 1

        res = differential_evolution(
            irt.negative_log_likelihood,
            bounds=[[self._lower_bound * 2, self._upper_bound * 2]],
            args=(response_vector, items[administered_items])
        )

        self._evaluations = res.nfev

        return res.x[0]

class BayesianEstimator(Estimator):
    """Estimator that uses Bayesian estimation for ability estimation

    :param prior_mean: mean of the prior distribution
    :param prior_sd: standard deviation of the prior distribution
    :param bounds: bounds of the ability estimate
    :param verbose: verbosity level of the estimation
    """

    def __str__(self):
        return 'Bayesian Estimator'

    def __init__(self,
            precision: int = 6,
            prior_mean: float = 0.,
            prior_sd: float = 1.,
            bounds: tuple = (-6, 6),
            verbose: bool = False
            ):
        super().__init__()
        self._precision = precision
        self._prior_mean = prior_mean
        self._prior_sd = prior_sd
        self._verbose = verbose
        self._evaluations = 0
        self._calls = 0
        self._bounds = bounds

    @property
    def calls(self) -> float:
        """How many times the estimator has been called to maximize/minimize the log-likelihood function

        :returns: number of times the estimator has been called to maximize/minimize the log-likelihood function"""
        return self._calls

    @property
    def evaluations(self) -> float:
        """Total number of times the estimator has evaluated the log-likelihood function during its existence

        :returns: number of function evaluations"""
        return self._evaluations

    @property
    def avg_evaluations(self) -> float:
        """Average number of function evaluations for all tests the estimator has been used

        :returns: average number of function evaluations"""
        return self._evaluations / self._calls

    def _bound_estimate(self, estimate):
        lbound, ubound = self._bounds
        return min(max(estimate, lbound), ubound)

    def estimate(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        response_vector: list = None,
        **kwargs
    ) -> float:
        """Returns the theta value that corresponds to the maximum a posteriori estimate, given the current state of the
         test for the given examinee. The posterior is obtained from summing the log-likelihood and the log of the normal
         density function.

        :param index: index of the current examinee in the simulator
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param response_vector: a boolean list containing the examinee's answers to the administered items
        :param est_theta: a float containing the current estimated proficiency
        :returns: the current theta estimate (based on the bounded MAP estimate)
        """
        if (index is None or self.simulator is None) and (
            items is None and administered_items is None or response_vector is None or
            est_theta is None
        ):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and response_vector is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            response_vector = self.simulator.response_vectors[index]
            est_theta = self.simulator.latest_estimations[index]

        self._calls += 1

        # need to constrain all estimates between these bounds, rather then, e.g.
        # min / max difficulties
        lower_bound, upper_bound = self._bounds

        best_theta = float('-inf')
        max_ll = float('-inf')

        self._evaluations = 0

        for _ in range(10):
            intervals = numpy.linspace(lower_bound, upper_bound, 10)
            if self._verbose:
                print(('Bounds: ' + str(lower_bound) + ' ' + str(upper_bound)))
                print(('Interval size: ' + str(intervals[1] - intervals[0])))

            for ii in intervals:
                self._evaluations += 1
                ll = irt.log_likelihood(ii, response_vector, items[administered_items]) + norm.logpdf(ii, loc = self._prior_mean, scale = self._prior_sd)
                if ll > max_ll:
                    max_ll = ll

                    if self._verbose:
                        print(
                            (
                                'Iteration: {0}, Theta: {1}, LL: {2}'.format(
                                    self._evaluations, ii, ll
                                )
                            )
                        )

                    if abs(best_theta - ii) < float('1e-' + str(self._precision)):
                        return self._bound_estimate(ii)

                    best_theta = ii

                else:
                    lower_bound = best_theta - (intervals[1] - intervals[0])
                    upper_bound = ii
                    # reset best_theta, in case optimum is to the left of it
                    max_ll = float('-inf')
                    break

        return self._bound_estimate(best_theta)
