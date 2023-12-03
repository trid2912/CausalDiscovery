import pandas as pd
import numpy as np
from math import log
class BaseEstimator(object):
    """
    Base class for estimators in pgmpy; `ParameterEstimator`,
    `StructureEstimator` and `StructureScore` derive from this class.

    Parameters
    ----------
    data: pandas DataFrame object
        object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, data=None, state_names=None):
        self.data = data
        # data can be None in the case when learning structure from
        # independence conditions. Look into PC.py.
        if self.data is not None:
            self.variables = list(data.columns.values)

            if not isinstance(state_names, dict):
                self.state_names = {
                    var: self._collect_state_names(var) for var in self.variables
                }
            else:
                self.state_names = dict()
                for var in self.variables:
                    if var in state_names:
                        if not set(self._collect_state_names(var)) <= set(
                            state_names[var]
                        ):
                            raise ValueError(
                                f"Data contains unexpected states for variable: {var}."
                            )
                        self.state_names[var] = state_names[var]
                    else:
                        self.state_names[var] = self._collect_state_names(var)

    def _collect_state_names(self, variable):
        "Return a list of states that the variable takes in the data."
        states = sorted(list(self.data.loc[:, variable].dropna().unique()))
        return states

    def state_counts(
        self,
        variable,
        parents=[],
        weighted=False,
        reindex=True,
    ):
        """
        Return counts how often each state of 'variable' occurred in the data.
        If a list of parents is provided, counting is done conditionally
        for each state configuration of the parents.

        Parameters
        ----------
        variable: string
            Name of the variable for which the state count is to be done.

        parents: list
            Optional list of variable parents, if conditional counting is desired.
            Order of parents in list is reflected in the returned DataFrame

        weighted: bool
            If True, data must have a `_weight` column specifying the weight of the
            datapoint (row). If False, each datapoint has a weight of `1`.

        reindex: bool
            If True, returns a data frame with all possible parents state combinations
            as the columns. If False, drops the state combinations which are not
            present in the data.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.estimators import BaseEstimator
        >>> data = pd.DataFrame(data={'A': ['a1', 'a1', 'a2'],
                                      'B': ['b1', 'b2', 'b1'],
                                      'C': ['c1', 'c1', 'c2']})
        >>> estimator = BaseEstimator(data)
        >>> estimator.state_counts('A')
            A
        a1  2
        a2  1
        >>> estimator.state_counts('C', parents=['A', 'B'])
        A  a1      a2
        B  b1  b2  b1  b2
        C
        c1  1   1   0   0
        c2  0   0   1   0
        >>> estimator.state_counts('C', parents=['A'])
        A    a1   a2
        C
        c1  2.0  0.0
        c2  0.0  1.0
        """
        parents = list(parents)

        if weighted and ("_weight" not in self.data.columns):
            raise ValueError("data must contain a `_weight` column if weighted=True")

        if not parents:
            # count how often each state of 'variable' occurred
            if weighted:
                state_count_data = self.data.groupby([variable])["_weight"].sum()
            else:
                state_count_data = self.data.loc[:, variable].value_counts()

            state_counts = (
                state_count_data.reindex(self.state_names[variable])
                .fillna(0)
                .to_frame()
            )

        else:
            parents_states = [self.state_names[parent] for parent in parents]
            # count how often each state of 'variable' occurred, conditional on parents' states
            if weighted:
                state_count_data = (
                    self.data.groupby([variable] + parents)["_weight"]
                    .sum()
                    .unstack(parents)
                )

            else:
                state_count_data = (
                    self.data.groupby([variable] + parents).size().unstack(parents)
                )

            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays(
                    [state_count_data.columns]
                )

            if reindex:
                # reindex rows & columns to sort them and to add missing ones
                # missing row    = some state of 'variable' did not occur in data
                # missing column = some state configuration of current 'variable's parents
                #                  did not occur in data
                row_index = self.state_names[variable]
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
                state_counts = state_count_data.reindex(
                    index=row_index, columns=column_index
                ).fillna(0)
            else:
                state_counts = state_count_data.fillna(0)

        return state_counts
    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        sample_size = len(self.data)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)
        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)
        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts
   
        score = np.sum(log_likelihoods)
        
        score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score

def is_dag(adjacency_matrix):


    for i in range(len(adjacency_matrix)):
        power = np.linalg.matrix_power(adjacency_matrix, i + 1)
        if np.trace(power) != 0:
            return False

    return True
def calculate_bic_score(graph, base_estimate):
    nodes = list(base_estimate.state_names.keys())

    num_nodes = len(nodes)
    bic_score = 0
    for k in range(num_nodes):
        parents = np.nonzero(graph[:,k])[0]
        parents = [nodes[k] for k in parents]
        if nodes[k] in parents:
            parents.remove(nodes[k])
        bic_score += base_estimate.local_score(nodes[k], parents)
    return bic_score