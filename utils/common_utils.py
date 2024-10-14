from enum import Enum, unique
import numpy as np
import json


@unique
class Complexity(Enum):
    """
    Enum class to represent the complexity of an algorithm
    """
    C = 1
    LOG_N = 2
    N = 3
    N_LOG_N = 4
    N_SQUARE = 5
    N_CUBE = 6
    N_POWER_4 = 7
    EXPONENTIAL = 8
    UNKNOWN = 9
    
    def __lt__(self, other):
        if isinstance(other, Complexity):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Complexity):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Complexity):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Complexity):
            return self.value >= other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Complexity):
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Complexity):
            return self.value != other.value
        return NotImplemented
    

@unique
class TaskType(Enum):
    ARITHMETIC = "arithmetic"
    NON_ARITHMETIC = "non-arithmetic"


def avg(lst):
    assert len(lst) > 0, "avg: List is empty"
    return sum(lst) / len(lst)


def geometric_mean(lst):
    if len(lst) == 0:
        return 0
    return np.exp(sum([np.log(x) for x in lst]) / len(lst))


def measure_id_ood_gap(id_perf, ood_perf):
    """
    Measure the gap between ID and OOD accuracy
    """
    gap = {}

    gap["diff"] = max(id_perf - ood_perf, 0)
    gap["ratio"] = ood_perf / id_perf if id_perf > 0 else 1
    gap["relative_diff"] = max(id_perf - ood_perf, 0) / id_perf if id_perf > 0 else 1
    
    return gap


def average_dict_list(dict_list):
    ret = {}
    for key, value_list in dict_list.items():
        assert len(value_list) > 0
        # value_list is a list of dictionaries
        avg_dict = {}
        representative_dict = value_list[0]
        for inner_key in representative_dict.keys():
            avg_dict[inner_key] = avg([d[inner_key] for d in value_list])
        ret[key] = avg_dict
    
    return ret