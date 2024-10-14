# Licensed under the MIT license.

import sys

sys.path.append(".")
from utils.common_utils import TaskType, Complexity
from utils.task_utils import *

import os, json, shutil
import numpy as np
from typing import List, Dict, Optional, final, Tuple, Union
from abc import ABC, abstractmethod
from tqdm import tqdm, trange
from itertools import permutations, combinations
import random
import string
from copy import copy, deepcopy
import heapq
from sympy import symbols, Poly
import math
from sympy import sympify
import re
from collections import defaultdict, Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations


num_workers = mp.cpu_count()


def process_item__generate_ood__nested_number_list(id_example, task, ood_elements, n_max_trials=50):
    n_trials = 0
    while True:
        ood_example = []
        id_example_outer_size = len(id_example)
        for i in range(id_example_outer_size):
            id_row = id_example[i]
            id_example_inner_size = len(id_row)
            ood_row = np.random.choice(ood_elements, id_example_inner_size, replace=True).tolist()
            ood_example.append(ood_row)

        if task.check_example(ood_example):
            return ood_example
        else:
            try:
                ood_example = task.try_transform_invalid_example(ood_example)
                return ood_example
            except:
                pass

        n_trials += 1
        if n_trials >= n_max_trials:
            print(f"Could not generate a valid OOD example for {task} on example {id_example}")
            return None


def process_item__generate_ood__number_list(id_example, task, ood_elements, n_max_trials=50):
    n_trials = 0
    id_example_size = len(id_example)
    while True:
        ood_example = np.random.choice(ood_elements, id_example_size, replace=True).tolist()

        if task.check_example(ood_example):
            return ood_example
        else:
            try:
                ood_example = task.try_transform_invalid_example(ood_example)
                return ood_example
            except:
                pass

        n_trials += 1
        if n_trials >= n_max_trials:
            print(f"Could not generate a valid OOD example for {task} on example {id_example}")
            return None


def process_item__generate_ood__number_list__many_duplicates(id_example, task, ood_elements, n_max_trials=50):
    n_trials = 0
    while True:
        id_element2cnt = Counter(id_example)
        ood_element2cnt = {}
        for _, cnt in id_element2cnt.items():
            ood_element = random.choice(ood_elements)
            ood_element2cnt[ood_element] = cnt

        ood_example = []
        for ood_element, cnt in ood_element2cnt.items():
            ood_example += [ood_element] * cnt

        random.shuffle(ood_example)

        if task.check_example(ood_example):
            return ood_example
        else:
            try:
                ood_example = task.try_transform_invalid_example(ood_example)
                return ood_example
            except:
                pass

        n_trials += 1
        if n_trials >= n_max_trials:
            print(f"Could not generate a valid OOD example for {task} on example {id_example}")
            return None


class _Task(ABC):
    def __init__(self) -> None:
        self.task_type: TaskType = None
        self.task_title: str = None  # only used at step 0
        self.task_description: str = None  # used at step 0 and 4
        self.example_type_for_gpt_extract: str = None  # only used at step 1
        self.answer_format_requirements: str = None  # only used at step 5

        self.worst_complexity: Complexity = None
        self.best_complexity: Complexity = None

        #! Answer format requirements
        self._answer_format_requirements__num = (
            'The answer should be a single number. For example, "The answer is 123" is in a valid format.'
        )
        self._answer_format_requirements__list = 'The answer should be a list. The list must be wrapped by square brackets and each element in the list must be a single number and must be separated by commas. For example, "The answer is [12, 34, 56]" is in a valid format.'
        self._answer_format_requirements__list__two_num = 'The answer should be a list of exactly two numbers. The list must be wrapped by square brackets and each element in the list must be a single number and must be separated by commas. For example, "The answer is [12, 34]" is in a valid format.'
        self._answer_format_requirements__list__three_num = 'The answer should be a list of exactly three numbers. The list must be wrapped by square brackets and each element in the list must be a single number and must be separated by commas. For example, "The answer is [12, 34, 56]" is in a valid format.'
        self._answer_format_requirements__list__four_num = 'The answer should be a list of exactly four numbers. The list must be wrapped by square brackets and each element in the list must be a single number and must be separated by commas. For example, "The answer is [12, 34, 56, 78]" is in a valid format.'

    def _sanity_check(self):
        #! Sanity check attributes
        assert self.task_type is not None, breakpoint()
        assert self.task_title is not None, breakpoint()
        assert self.task_description is not None, breakpoint()
        assert self.example_type_for_gpt_extract is not None, breakpoint()
        assert self.answer_format_requirements is not None, breakpoint()
        assert self.worst_complexity is not None, breakpoint()
        assert self.best_complexity is not None, breakpoint()

    # ************ Pipeline ************

    def _prepare_input(self, whatever_input):
        return whatever_input

    @abstractmethod
    def _execute(self, processed_input):
        raise NotImplementedError

    def _prepare_output(self, execution_output):
        return execution_output

    @final
    def _run(self, whatever_input):
        processed_input = self._prepare_input(whatever_input)
        execution_outputs = self._execute(processed_input)
        assert (
            isinstance(execution_outputs, tuple) and len(execution_outputs) > 0
        ), f"Execution outputs: {execution_outputs} is not a tuple or is empty."
        processed_outputs = ()
        for exec_o in execution_outputs:
            processed_o = self._prepare_output(exec_o)
            processed_outputs += (processed_o,)
        return processed_input, processed_outputs

    @final
    def make_io_pair(self, example) -> Tuple[str, str]:
        self._sanity_check()

        assert self.check_example(example)
        i, o_tuple = self._run(example)
        i_str = self._cast_input_to_string(deepcopy(i))
        o_str_tuple = ()
        for o in o_tuple:
            o_str = self._cast_output_to_string(deepcopy(o))
            o_str_tuple += (o_str,)
        return i, o_tuple, i_str, o_str_tuple

    # ************ Test Inputs generation ************
    @final
    def _random_test_inputs_prompt__NumList(self):
        assert self.task_title is not None, f"Task title is not set for task {self}."
        assert self.task_description is not None, f"Task description is not set for task {self}."

        return f"Randomly generate number lists as test inputs for testing a program written for the task of {self.task_title}. The task description is: {self.task_description} Enclose each number list by square brackets, e.g. [x1, x2, x3, x4]. Do not generate the corresponding output. Do not use or generate any code. Make sure that the number elements are non-negative integers. Make sure that the number lists are not empty and not too long.\n\nNow please generate as many such number lists as possible:\n\n"

    @final
    def _random_test_inputs_prompt__NumListPair(self):
        assert self.task_title is not None, f"Task title is not set for task {self}."
        assert self.task_description is not None, f"Task description is not set for task {self}."

        return f"Randomly generate pairs of number lists as test inputs for testing a program written for the task of {self.task_title}. The task description is: {self.task_description} Enclose each of the two number list by square brackets and put them in another list, e.g. [[x1, x2], [x3, x4]]. Do not generate the corresponding output. Do not use or generate any code. Make sure that the number elements are non-negative integers. Make sure that the number lists are not empty and not too long.\n\nNow please generate as many such pairs of number lists as possible:\n\n"

    @abstractmethod
    def random_test_inputs_prompt(self):
        raise NotImplementedError

    # ************ Casters ************
    @final
    def _cast_number_to_string(self, obj: Union[float, int]) -> str:
        return str(obj)

    @final
    def _cast_list_to_string(self, obj: List) -> str:
        assert len(obj) > 0
        for i in range(len(obj)):
            obj[i] = str(obj[i])
        return " ".join(obj)

    @final
    def _cast_list_to_string_and_tuple(self, obj: Tuple[List[int], Tuple[int, int]]) -> int:
        list_input, double_input = obj
        first, second = double_input
        assert len(list_input) > 0
        assert isinstance(first, int)
        assert isinstance(second, int)
        for i in range(len(list_input)):
            list_input[i] = str(list_input[i])

        return "$l$ = " + " ".join(list_input) + ", $a$ = " + str(first) + ", $b$ = " + str(second)

    @final
    def _cast_list_to_poly(self, obj: List[int]) -> str:
        assert len(obj) > 0

        terms = []
        degree = len(obj) - 1

        for i, coef in enumerate(obj):
            if coef == 0:
                continue

            power = degree - i
            if power == 0:
                term = f"{coef}"
            elif power == 1:
                term = f"{coef}x" if coef != 1 else "x"
            else:
                term = f"{coef}x^{power}" if coef != 1 else f"x^{power}"

            if terms:
                if coef > 0:
                    terms.append(f" + {term}")
                else:
                    terms.append(f"{term}")
            else:
                terms.append(term)

        assert len(terms) > 0
        ret = "".join(terms)
        ret = "$" + ret + "$"

        return ret

    @final
    def _cast_list_to_string_for_strplusint(self, obj: Tuple[List[int], int]) -> str:
        list_input, single_input = obj

        assert len(list_input) > 0, "List must not be empty"
        assert isinstance(single_input, int), "Single input must be an integer"

        for i in range(len(list_input)):
            list_input[i] = str(list_input[i])

        return "$l$ = " + " ".join(list_input) + ", $k$ = " + str(single_input)

    @final
    def _cast_list_to_string_for_str_twins(self, obj: List[str]) -> str:
        assert len(obj) == 2
        assert all(isinstance(e, str) for e in obj)
        return "$s_1$ = " + obj[0] + ", $s_2$ = " + obj[1]

    @final
    def _cast_matrix_to_string(self, obj: List[List], matrix_type: str = "latex") -> str:
        if matrix_type == "latex":
            latex_matrix = "\\begin{bmatrix}\n"
            rows = []
            for row in obj:
                latex_row = " & ".join(map(str, row))
                rows.append(latex_row)
            latex_matrix += " \\\\\n".join(rows)
            latex_matrix += "\n\\end{bmatrix}"

            latex_document = "\\[\n" f"{latex_matrix}\n" "\\]\n"
            return latex_document
        elif matrix_type == "nested_list":
            raise NotImplementedError

    @final
    def _cast_matrix_to_string_for_matrixwithsize(self, obj: List[List[int]]) -> str:
        n_str = str(len(obj))
        matrix_str = self._cast_matrix_to_string(obj)
        return f"$n$ = {n_str}, $D$ =\n{matrix_str}"

    @abstractmethod
    def _cast_input_to_string(self, input):
        raise NotImplementedError

    @abstractmethod
    def _cast_output_to_string(self, output):
        raise NotImplementedError

    # ************ OOD data generator ************

    @final
    def _generate_ood_data__non_ascii_string(
        self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path
    ):
        assert False, f"[Task.py:_generate_ood_data__non_ascii_string] ==> Not supported yet."

    @final
    def _generate_ood_data__number_list(
        self,
        id_elements_path,
        id_examples_path,
        ood_elements_save_path,
        ood_examples_save_path,
        number_range,
        nested=False,
        n_max_trials=50,
    ):
        with open(id_elements_path, "r") as f:
            id_elements = json.load(f)

        unique_id_elements = list(set(id_elements))

        unique_ood_elements = list(set(list(range(number_range[0], number_range[1] + 1))) - set(unique_id_elements))
        assert len(unique_ood_elements) > 0

        ood_elements = np.random.choice(unique_ood_elements, len(id_elements), replace=True).tolist()

        with open(id_examples_path, "r") as f:
            id_examples = json.load(f)
        
        suitable_size = 256 * 100
        if len(id_examples) < suitable_size:
            print(f"[Task.py:_generate_ood_data__number_list] ==> ID examples data is too small: {len(id_examples)}")
            id_examples = id_examples * (suitable_size // len(id_examples) + 1)
            print(f"[Task.py:_generate_ood_data__number_list] ==> ID examples data replicated to: {len(id_examples)}")

        ood_examples = []

        if nested:
            assert all(isinstance(e, list) for e in id_examples)
            for example in id_examples:
                for row in example:
                    assert isinstance(row, list)
                    for element in row:
                        assert not isinstance(
                            element, list
                        ), f"Nested list of more than 2 levels is not supported: {example}"

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_item__generate_ood__nested_number_list, id_example, self, ood_elements, n_max_trials
                    )
                    for id_example in id_examples
                ]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    ood_examples.append(future.result())
        else:
            for example in id_examples:
                for element in example:
                    assert not isinstance(element, list), f"Non-nested mode but element is a list: {element}"

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_item__generate_ood__number_list, id_example, self, ood_elements, n_max_trials
                    )
                    for id_example in id_examples
                ]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    ood_examples.append(future.result())

        assert len(ood_examples) == len(id_examples)

        ood_examples = [format_element_or_example(e, self) for e in ood_examples if self.check_example(e)]
        ood_elements = [get_elements_from_example(e, self) for e in ood_examples]

        # Deduplicate
        print(f"[Task.py:_generate_ood_data__number_list] ==> Deduplicating OOD data for task {self}...")
        all_ood_examples_str_list = [json.dumps(x) for x in ood_examples]
        all_ood_examples_str_list = list(set(all_ood_examples_str_list))
        ood_examples = [json.loads(x) for x in all_ood_examples_str_list]
        print(
            f"[Task.py:_generate_ood_data__number_list] ==> Deduplicated to {len(ood_examples)} OOD examples for task {self}."
        )

        print(f"[Task.py:_generate_ood_data__number_list] ==> Saving OOD elements to {ood_elements_save_path}")
        with open(ood_elements_save_path, "w") as f:
            json.dump(ood_elements, f, ensure_ascii=True, cls=NpEncoder)
        print(f"[Task.py:_generate_ood_data__number_list] ==> Saving OOD examples to {ood_examples_save_path}")
        with open(ood_examples_save_path, "w") as f:
            json.dump(ood_examples, f, ensure_ascii=True, cls=NpEncoder)

    @final
    def _generate_ood_data__adjacency_matrix(
        self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path, number_range
    ):
        with open(id_elements_path, "r") as f:
            id_elements = json.load(f)

        unique_id_elements = list(set(id_elements))

        unique_ood_elements = list(set(list(range(number_range[0], number_range[1] + 1))) - set(unique_id_elements))

        ood_elements = np.random.choice(unique_ood_elements, len(id_elements), replace=True).tolist()

        with open(id_examples_path, "r") as f:
            id_examples = json.load(f)

        id_examples_sizes = [len(example) for example in id_examples]
        ood_examples = []
        for size in tqdm(id_examples_sizes):
            n_max_trials = 10
            n_trials = 0
            while True:
                ood_example = generate_adjacency_matrix(size, ood_elements)

                if self.check_example(ood_example):
                    ood_examples.append(ood_example)
                    break
                else:
                    try:
                        ood_example = self.try_transform_invalid_example(ood_example)
                        ood_examples.append(ood_example)
                        break
                    except:
                        pass

                n_trials += 1
                if n_trials >= n_max_trials:
                    breakpoint()

        assert len(ood_examples) == len(id_examples)

        ood_examples = [e for e in ood_examples if self.check_example(e)]
        ood_elements = [get_elements_from_example(e, self) for e in ood_examples]

        # Deduplicate
        print(f"[Task.py:_generate_ood_data__adjacency_matrix] ==> Deduplicating OOD data for task {self}...")
        all_ood_examples_str_list = [json.dumps(x, ensure_ascii=True, cls=NpEncoder) for x in ood_examples]
        all_ood_examples_str_list = list(set(all_ood_examples_str_list))
        ood_examples = [json.loads(x) for x in all_ood_examples_str_list]
        print(
            f"[Task.py:_generate_ood_data__adjacency_matrix] ==> Deduplicated to {len(ood_examples)} OOD examples for task {self}."
        )

        print(f"[Task.py:_generate_ood_data__adjacency_matrix] ==> Saving OOD elements to {ood_elements_save_path}")
        with open(ood_elements_save_path, "w") as f:
            json.dump(ood_elements, f, ensure_ascii=True, cls=NpEncoder)
        print(f"[Task.py:_generate_ood_data__adjacency_matrix] ==> Saving OOD examples to {ood_examples_save_path}")
        with open(ood_examples_save_path, "w") as f:
            json.dump(ood_examples, f, ensure_ascii=True, cls=NpEncoder)

    @final
    def _generate_ood_data__non_ascii_string_list(
        self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path
    ):
        assert False, f"[Task.py:_generate_ood_data__non_ascii_string_list] ==> Not supported yet."

    @abstractmethod
    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        raise NotImplementedError

    @final
    def generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        assert os.path.exists(id_elements_path) and os.path.exists(
            id_examples_path
        ), f"ID data for task {self} does not exist."

        tgt_dirs = [os.path.dirname(ood_elements_save_path), os.path.dirname(ood_examples_save_path)]
        for tgt_dir in tgt_dirs:
            os.makedirs(tgt_dir, exist_ok=True)

        self._generate_ood_data(id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path)

    # ************ Element checkers ************

    @final
    def _check_element__string(self, element):
        return isinstance(element, str) and 0 < len(element) <= 45 and " " not in element

    @final
    def _check_element__non_neg_int(self, element):
        if not isinstance(element, int) and not isinstance(element, np.int64):
            return False

        if element < 0:
            return False  # We only consider non-negative numbers now

        bool_a = element == element and element != float("inf") and element != float("-inf")
        try:
            int(element)  # Check if the number is an integer
            bool_b = True
        except:
            bool_b = False

        return bool_a and bool_b

    @final
    def _check_element__char(self, char) -> bool:
        if not isinstance(char, str):
            return False

        if len(char) != 1:
            return False

        return True

    @abstractmethod
    def check_element(self, element):
        raise NotImplementedError

    # ************ Example checkers ************

    @final
    def _check_example__adjacency_matrix(self, example):
        try:
            # Check if the outer structure is a list
            assert isinstance(example, list)

            # Check if the outer list is not empty
            assert len(example) > 0

            # Check if all elements in the outer list are lists
            assert all(isinstance(row, list) for row in example)

            # Check if all elements are integers
            for row in example:
                for element in row:
                    try:
                        assert self.check_element(element)
                    except:
                        raise ValueError(f"Element {element} is not an integer")

            # Check if the matrix is square (number of rows should be equal to number of columns)
            num_rows = len(example)
            assert all(len(row) == num_rows for row in example)

            # Check if the diagonal elements are all 0
            for i in range(num_rows):
                assert example[i][i] == 0

            # Check if the matrix is symmetric
            for i in range(num_rows):
                for j in range(i + 1, num_rows):
                    if example[i][j] != example[j][i]:
                        raise ValueError("Matrix is not symmetric")

            # Check if all elements in the inner lists are integers
            for row in example:
                for element in row:
                    try:
                        assert self.check_element(element)
                    except:
                        self.check_element(element)

            return True
        except:
            return False

    @final
    def _check_example__list(self, example):
        try:
            assert isinstance(example, list)
            assert len(example) > 0
            assert all(not isinstance(e, list) for e in example)
            if all(self.check_element(element) for element in example):
                return True
            else:
                return False
        except:
            return False

    @final
    def _check_example__list_pair(self, example):
        if not isinstance(example, list):
            return False

        if not len(example) == 2:
            return False

        bool_a = self._check_example__list(example[0])
        bool_b = self._check_example__list(example[1])

        return bool_a and bool_b

    @final
    def _check_example__string(self, example) -> bool:
        if not isinstance(example, str):
            return False

        if len(example) == 0:
            return False

        for char in example:
            if not self._check_element__char(char):
                return False

        return True

    @abstractmethod
    def check_example(self, example):
        raise NotImplementedError

    # ************ Try transform ************
    @final
    def _try_transform_invalid_example__adjacency_matrix(self, invalid_example, min_dim=0, max_dim=math.inf):
        # First, make the list of lists to be a square matrix (cut off the extra rows or columns)
        assert isinstance(invalid_example, list)
        assert min_dim <= len(invalid_example) <= max_dim

        elements = get_elements_from_example(invalid_example, self)
        dim = len(invalid_example)

        adjacency_matrix = [[0 for _ in range(dim)] for _ in range(dim)]

        for row in range(dim):
            for col in range(row + 1, dim):
                if row == col:
                    continue
                adjacency_matrix[row][col] = np.random.choice(elements)
                adjacency_matrix[col][row] = adjacency_matrix[row][col]

        return adjacency_matrix

    def _try_transform_invalid_example(self, invalid_example):
        assert False

    @final
    def try_transform_invalid_example(self, invalid_example):
        valid_example = self._try_transform_invalid_example(invalid_example)
        assert self.check_example(
            valid_example
        ), f"Invalid example: {invalid_example}, transformed to: {valid_example}, but still invalid."
        return valid_example

    # ************ Element formatters ************

    @final
    def _format_element__int(self, element):
        return int(element)

    @final
    def _format_element__string(self, element):
        return str(element).replace(" ", "")

    @abstractmethod
    def format_element(self, element):
        raise NotImplementedError

    # ************ Answer extractors ************

    @final
    def _extract_answer__single_number(self, model_output: str) -> str:
        cleaned_model_output = self._clean_model_output(
            model_output, chars_to_remove=["```json", "```python", "`", "\n", '"']
        )
        extract_result = extract_number_after_keyword(cleaned_model_output)

        if extract_result is None:
            return None

        return extract_result

    @final
    def _extract_answer__string(self, model_output: str) -> str:
        cleaned_model_output = self._clean_model_output(model_output, chars_to_remove=["`", "\n"])
        extract_result = extract_string_after_keyword(cleaned_model_output)

        if extract_result is None:
            return None

        return extract_result

    @final
    def _extract_answer__list(self, model_output: str) -> str:
        cleaned_model_output = self._clean_model_output(
            model_output, chars_to_remove=["```json", "```python", "`", "\n", '"']
        )
        extract_result = extract_number_list_after_keyword(cleaned_model_output)

        if extract_result is None:
            return None

        if "{" in extract_result or "}" in extract_result:
            #! IMPORTANT: This is a temporary fix. The model should not output a JSON object with dictionaries.
            return None

        return extract_result

    @final
    def _extract_answer__matrix(self, model_output: str) -> str:
        # todo: check
        cleaned_model_output = self._clean_model_output(
            model_output, chars_to_remove=["```json", "```python", "`", "\n", '"', "{", "}"]
        )
        extract_result = extract_number_list_after_keyword(cleaned_model_output)

        if extract_result is None:
            return None

        if "{" in extract_result or "}" in extract_result:
            #! IMPORTANT: This is a temporary fix. The model should not output a JSON object with dictionaries.
            return None

        return extract_result

    @abstractmethod
    def extract_answer(self, model_output: str) -> str:
        raise NotImplementedError

    # ************ Evaluators ************

    @final
    def _parse_js(self, model_answer_js: Optional[str], desired_output_js: str):
        try:
            model_answer = json.loads(model_answer_js)
            desired_output = json.loads(desired_output_js)
            return model_answer, desired_output
        except json.JSONDecodeError:
            breakpoint()

    @final
    def _evaluate__single_number(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        assert isinstance(desired_output_js, str)
        assert len(desired_output_js) > 0

        if not isinstance(model_answer_js, str):
            return False

        if len(model_answer_js) == 0:
            return False

        model_answer, desired_output = self._parse_js(model_answer_js, desired_output_js)

        try:
            model_answer = int(float(model_answer))
            desired_output = int(float(desired_output))
            return model_answer == desired_output
        except:
            return False

    @final
    def _evaluate__string(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        assert isinstance(desired_output_js, str)
        assert len(desired_output_js) > 0

        if not isinstance(model_answer_js, str):
            return False

        if len(model_answer_js) == 0:
            return False

        model_answer, desired_output = self._parse_js(model_answer_js, desired_output_js)

        return model_answer == desired_output

    @final
    def _evaluate__string_list(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        assert isinstance(desired_output_js, str)
        assert len(desired_output_js) > 0

        if not isinstance(model_answer_js, str):
            return False

        if len(model_answer_js) == 0:
            return False

        model_answer, desired_output = self._parse_js(model_answer_js, desired_output_js)

        if model_answer is None:
            return False

        if len(model_answer) != len(desired_output):
            return False

        return all(model_answer[i] == desired_output[i] for i in range(len(model_answer)))

    @final
    def _evaluate__number_list(
        self,
        model_answer_js: Optional[str],
        desired_output_js: str,
        unordered_ok: bool = False,
        reverse_ok: bool = False,
    ) -> bool:
        assert isinstance(desired_output_js, str)
        assert len(desired_output_js) > 0
        assert sum([unordered_ok, reverse_ok]) <= 1, f"Only one of unordered_ok and reverse_ok can be True."

        if not isinstance(model_answer_js, str):
            return False

        if len(model_answer_js) == 0:
            return False

        model_answer, desired_output = self._parse_js(model_answer_js, desired_output_js)

        assert isinstance(model_answer, list), breakpoint()
        assert isinstance(desired_output, list), breakpoint()

        assert model_answer is not None and desired_output is not None, breakpoint()

        if len(model_answer) != len(desired_output):
            return False

        try:
            model_answer_arr = np.array(model_answer, dtype=float)
            desired_output_arr = np.array(desired_output, dtype=float)
        except:
            breakpoint()

        if unordered_ok:
            model_answer_arr = np.sort(model_answer_arr)
            desired_output_arr = np.sort(desired_output_arr)

        if reverse_ok:
            return np.allclose(model_answer_arr, desired_output_arr) or np.allclose(
                model_answer_arr, desired_output_arr[::-1]
            )
        else:
            return np.allclose(model_answer_arr, desired_output_arr)

    @final
    def _evaluate__number_matrix(
        self,
        model_answer_js: Optional[str],
        desired_output_js: str,
        unordered_ok: bool = False,
        reverse_ok: bool = False,
    ) -> bool:
        # todo: check 
        assert isinstance(desired_output_js, str)
        assert len(desired_output_js) > 0
        assert sum([unordered_ok, reverse_ok]) <= 1, "Only one of unordered_ok and reverse_ok can be True."

        if not isinstance(model_answer_js, str) or len(model_answer_js) == 0:
            return False

        model_answer, desired_output = self._parse_js(model_answer_js, desired_output_js)

        # Convert lists of lists to NumPy arrays
        try:
            model_answer_arr = np.array(model_answer, dtype=float)
            desired_output_arr = np.array(desired_output, dtype=float)
        except:
            # Handle conversion issues
            return False

        # Check matrix dimensions
        if model_answer_arr.shape != desired_output_arr.shape:
            return False

        if unordered_ok:
            # Sort rows and columns to compare matrices regardless of order
            model_answer_arr = np.sort(model_answer_arr, axis=0)  # Sort columns
            desired_output_arr = np.sort(desired_output_arr, axis=0)  # Sort columns
            model_answer_arr = np.sort(model_answer_arr, axis=1)  # Sort rows
            desired_output_arr = np.sort(desired_output_arr, axis=1)  # Sort rows

        if reverse_ok:
            # Check if the matrix or its reverse matches
            return np.allclose(model_answer_arr, desired_output_arr) or np.allclose(
                model_answer_arr, np.flip(desired_output_arr, axis=0)
            )
        else:
            return np.allclose(model_answer_arr, desired_output_arr)

    @abstractmethod
    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        raise NotImplementedError

    # ************ Others ************

    @final
    def _clean_model_output(self, model_output: str, chars_to_remove: List[str]):
        cleaned_model_output = model_output
        for char in chars_to_remove:
            cleaned_model_output = cleaned_model_output.replace(char, "")
        return cleaned_model_output

    @final
    def _make_json(self, input_or_output):
        """Only used in 3_get_io_pairs.py"""
        try:
            return json.dumps(input_or_output, ensure_ascii=False)
        except:
            breakpoint()

    def make_input_json(self, input):
        return self._make_json(input)

    def make_output_json(self, output):
        return self._make_json(output)

    def __str__(self) -> str:
        return self.__class__.__name__


class _Task_NumList_2_Num(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.example_type_for_gpt_extract = "list"
        self.answer_format_requirements = self._answer_format_requirements__num

    def random_test_inputs_prompt(self):
        return self._random_test_inputs_prompt__NumList()

    def _cast_input_to_string(self, input: List[int]) -> str:
        return self._cast_list_to_string(input)

    def _cast_output_to_string(self, output: int) -> str:
        return self._cast_number_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__number_list(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(1_000, 9_999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__single_number(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__single_number(model_answer_js, desired_output_js)


class _Task_NumList_2_NumList(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.example_type_for_gpt_extract = "list"
        self.answer_format_requirements = self._answer_format_requirements__list

    def random_test_inputs_prompt(self):
        return self._random_test_inputs_prompt__NumList()

    def _cast_input_to_string(self, input: List[int]) -> str:
        return self._cast_list_to_string(input)

    def _cast_output_to_string(self, output: List[int]) -> str:
        return self._cast_list_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__number_list(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(1_000, 9_999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__list(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js)


class _Task_NumList_Num_2_Num(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.example_type_for_gpt_extract = "list"
        self.answer_format_requirements = self._answer_format_requirements__num

    def random_test_inputs_prompt(self):
        return self._random_test_inputs_prompt__NumList()

    def _cast_input_to_string(self, input: Tuple[List[int], int]) -> str:
        return self._cast_list_to_string_for_strplusint(input)

    def _cast_output_to_string(self, output: int) -> str:
        return self._cast_number_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__number_list(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(1_000, 9_999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__single_number(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__single_number(model_answer_js, desired_output_js)


class _Task_NumList_NumPair_2_Num(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.example_type_for_gpt_extract = "list"
        self.answer_format_requirements = self._answer_format_requirements__num

    def random_test_inputs_prompt(self):
        return self._random_test_inputs_prompt__NumList()

    def _cast_input_to_string(self, input: Tuple[List[int], Tuple[int, int]]) -> str:
        return self._cast_list_to_string_and_tuple(input)

    def _cast_output_to_string(self, output: int) -> str:
        return self._cast_number_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__number_list(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(1_000, 9_999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__single_number(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__single_number(model_answer_js, desired_output_js)


class _Task_NumList_Num_2_NumList(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.example_type_for_gpt_extract = "list"
        self.answer_format_requirements = self._answer_format_requirements__list

    def random_test_inputs_prompt(self):
        return self._random_test_inputs_prompt__NumList()

    def _cast_input_to_string(self, input: Tuple[List[int], int]) -> str:
        return self._cast_list_to_string_for_strplusint(input)

    def _cast_output_to_string(self, output: List[int]) -> str:
        return self._cast_list_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__number_list(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(1_000, 9_999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__list(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js)


class _Task_NumList_NumPair_2_NumList(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.example_type_for_gpt_extract = "list"
        self.answer_format_requirements = self._answer_format_requirements__list

    def random_test_inputs_prompt(self):
        return self._random_test_inputs_prompt__NumList()

    def _cast_input_to_string(self, input: Tuple[List[int], Tuple[int, int]]) -> str:
        return self._cast_list_to_string_and_tuple(input)

    def _cast_output_to_string(self, output: List[int]) -> str:
        return self._cast_list_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__number_list(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(1_000, 9_999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__list(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js)


class _Task_NumListPair_2_NumList(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.example_type_for_gpt_extract = "list"
        self.answer_format_requirements = self._answer_format_requirements__list

    def random_test_inputs_prompt(self):
        return self._random_test_inputs_prompt__NumListPair()

    def _cast_input_to_string(self, input: List[List[int]]) -> str:
        input_1, input_2 = input
        return "$l_1$ = " + self._cast_list_to_string(input_1) + ", $l_2$ = " + self._cast_list_to_string(input_2)

    def _cast_output_to_string(self, output: List[int]) -> str:
        return self._cast_list_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__number_list(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(1_000, 9_999),
            nested=True,
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__list(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js)


#! Done
class FindMinimum(_Task_NumList_2_Num):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding minimum"
        self.task_description = "Given a list of numbers separated by spaces, find the smallest number."
        self.worst_complexity = Complexity.N
        self.best_complexity = Complexity.N

    def _execute(self, processed_input: List[int]):
        return (min(processed_input),)

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            return len(example) >= 2
        return False


#! Done
class FindMaximum(_Task_NumList_2_Num):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding maximum"
        self.task_description = "Given a list of numbers separated by spaces, find the largest number."
        self.worst_complexity = Complexity.N
        self.best_complexity = Complexity.N

    def _execute(self, processed_input: List[int]):
        return (max(processed_input),)

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            return len(example) >= 2
        return False


#! Done
class FindMode(_Task_NumList_2_Num):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding mode"
        self.task_description = "Given a list of numbers separated by spaces, find the mode of the numbers."
        self.worst_complexity = Complexity.UNKNOWN
        self.best_complexity = Complexity.N

    def _find_modes(self, processed_input: List[int]):
        element2num = {}
        for element in processed_input:
            if element in element2num:
                element2num[element] += 1
            else:
                element2num[element] = 1
        num2elements = {}
        for element, num in element2num.items():
            if num in num2elements:
                num2elements[num].append(element)
            else:
                num2elements[num] = [element]

        max_num = max(num2elements.keys())
        return num2elements[max_num]

    def _execute(self, processed_input: List[int]):
        modes = self._find_modes(processed_input)
        assert len(modes) == 1
        return (modes[0],)

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 2
            if bool_b:
                return len(self._find_modes(example)) == 1

        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 2

        # Find the modes
        modes = self._find_modes(invalid_example)
        # Select a random mode
        mode = random.choice(modes)
        # Add some more of it
        invalid_example.extend([mode] * random.randint(2, 8))
        random.shuffle(invalid_example)

        return invalid_example


#! Done
class FindTopk(_Task_NumList_Num_2_Num):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding top k"
        self.task_description = (
            "Given a list of numbers $l$ separated by spaces and a positive integer $k$, find the $k$th largest number."
        )
        self.worst_complexity = Complexity.N_SQUARE
        self.best_complexity = Complexity.N

    def _prepare_input(self, whatever_input):
        k = random.randint(2, len(whatever_input) // 2)
        return (whatever_input, k)

    def _findtopk(self, processed_input_tuple):
        processed_input, k = processed_input_tuple
        assert 1 <= k <= len(processed_input), breakpoint()

        # Sort the list in descending order and take the top k elements
        sorted_input = sorted(processed_input, reverse=True)
        return sorted_input[k - 1]

    def _execute(self, processed_input_tuple: Tuple[List[int], int]):
        return (self._findtopk(processed_input_tuple),)

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            return len(example) >= 4
        return False


#! Done
class SortNumbers(_Task_NumList_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "sorting numbers"
        self.task_description = "Given a list of numbers separated by spaces, sort the numbers in ascending order."
        self.worst_complexity = Complexity.N_SQUARE
        self.best_complexity = Complexity.N

    def _execute(self, processed_input: List[int]):
        return (sorted(processed_input),)

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            return len(example) >= 2
        return False


#! Done
class RemoveDuplicateNumbers(_Task_NumList_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "removing duplicate numbers"
        self.task_description = "Given a list of numbers separated by spaces, remove duplicate numbers so that every number appears only once, and output the remaining numbers in their original order."
        self.worst_complexity = Complexity.N_SQUARE
        self.best_complexity = Complexity.N

    def _execute(self, processed_input: List[int]) -> List[int]:
        seen = set()
        result = []

        for num in processed_input:
            if num not in seen:
                seen.add(num)
                result.append(num)

        return (result,)

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            return len(example) >= 2
        return False

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        number_range = (1_000, 9_999)

        with open(id_elements_path, "r") as f:
            id_elements = json.load(f)

        unique_id_elements = list(set(id_elements))

        unique_ood_elements = list(set(list(range(number_range[0], number_range[1] + 1))) - set(unique_id_elements))
        assert len(unique_ood_elements) > 0

        ood_elements = np.random.choice(unique_ood_elements, len(id_elements), replace=True).tolist()

        with open(id_examples_path, "r") as f:
            id_examples = json.load(f)

        ood_examples = []

        for example in id_examples:
            for element in example:
                assert not isinstance(element, list), f"Non-nested mode but element is a list: {element}"

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_item__generate_ood__number_list__many_duplicates, id_example, self, ood_elements
                )
                for id_example in id_examples
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                ood_examples.append(future.result())

        assert len(ood_examples) == len(id_examples)

        ood_examples = [format_element_or_example(e, self) for e in ood_examples if self.check_example(e)]
        ood_elements = [get_elements_from_example(e, self) for e in ood_examples]

        print(f"[Task.py:_generate_ood_data__number_list] ==> Saving OOD elements to {ood_elements_save_path}")
        with open(ood_elements_save_path, "w") as f:
            json.dump(ood_elements, f, ensure_ascii=True, cls=NpEncoder)
        print(f"[Task.py:_generate_ood_data__number_list] ==> Saving OOD examples to {ood_examples_save_path}")
        with open(ood_examples_save_path, "w") as f:
            json.dump(ood_examples, f, ensure_ascii=True, cls=NpEncoder)


#! Done
class LongestIncreasingSubsequence(_Task_NumList_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding the longest increasing subsequence"
        self.task_description = "Given a list of numbers separated by spaces, return the longest strictly increasing subsequence. A subsequence is a list that can be derived from another list by deleting some or no elements without changing the order of the remaining elements."
        self.worst_complexity = Complexity.N_LOG_N
        self.best_complexity = Complexity.N_SQUARE

    def _find_longest_increasing_subsequences(self, nums: List[int]) -> List[int]:
        if not nums:
            return []

        n = len(nums)
        dp = [1] * n  # DP array initialized with 1
        prev = [-1] * n  # To store the index of the previous element in the LIS

        # Fill DP table
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
                    prev[i] = j

        # Find the index of the maximum value in dp array
        max_length = max(dp)
        index = dp.index(max_length)

        # Reconstruct the LIS
        lis = []
        while index != -1:
            lis.append(nums[index])
            index = prev[index]

        return lis[::-1]  # Reverse the result to get the correct order

    def _execute(self, nums: List[int]) -> List[int]:
        longest_increasing_subsequences = self._find_longest_increasing_subsequences(nums)
        if not longest_increasing_subsequences:
            raise ValueError("No strictly increasing subsequence found")
        return (longest_increasing_subsequences,)

    def _check_valid(self, example):
        longest_sequence = self._find_longest_increasing_subsequences(example)
        return len(longest_sequence) > 1

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 2
            if bool_b:
                return self._check_valid(example)

        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 2

        n_max_trials = 10
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py] ==> This example is invalid: {invalid_example}")

        return invalid_example


#! Done
class LongestConsecutiveElements(_Task_NumList_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding the longest consecutive elements"
        self.task_description = "Given a list of numbers separated by spaces, return the longest consecutive number sequence in ascending order. A consecutive sequence is a sequence of numbers where each number is exactly 1 greater than the previous number."
        self.worst_complexity = Complexity.EXPONENTIAL
        self.best_complexity = Complexity.N_SQUARE

    def _find_longest_sequences(self, processed_input_tuple: List[int]) -> List[int]:
        if not processed_input_tuple:
            raise []

        nums = set(processed_input_tuple)
        longest_streak = 0
        longest_start = num = None

        for num in nums:
            # Check if it's the start of a sequence
            if num - 1 not in nums:
                current_num = num
                current_streak = 1
                start_num = current_num

                # Count the streak length
                while current_num + 1 in nums:
                    current_num += 1
                    current_streak += 1

                # Update longest streak if necessary
                if current_streak > longest_streak:
                    longest_streak = current_streak
                    longest_start = start_num

        # Construct the longest sequence list
        if longest_start is not None:
            return list(range(longest_start, longest_start + longest_streak))
        else:
            raise []

    def _execute(self, processed_input_tuple: List[int]) -> List[int]:
        if not processed_input_tuple:
            raise ValueError("Input list is empty")

        longest_sequences = self._find_longest_sequences(processed_input_tuple)

        if not longest_sequences:
            raise ValueError("No consecutive sequence found")

        return (longest_sequences,)

    def _check_valid(self, example):
        longest_sequence = self._find_longest_sequences(example)
        return len(longest_sequence) > 1

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 2
            if bool_b:
                return self._check_valid(example)

        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 2

        elements = set(invalid_example)

        n_max_trials = 10
        n = 0
        while not self._check_valid(invalid_example):
            n += 1

            #! according to the elements, we try to make the example have a longer consecutive sequence
            elements = np.random.choice(invalid_example, len(invalid_example) // 2, replace=False)
            for e in elements:
                if e + 1 not in invalid_example:
                    invalid_example.append(e + 1)
                if e - 1 not in invalid_example:
                    invalid_example.append(e - 1)

            if n == n_max_trials:
                raise ValueError(f"[Task.py:LongestConsecutiveElements] ==> This example is invalid: {invalid_example}")

        return invalid_example


#! Done
class TwoSum(_Task_NumList_Num_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding two numbers adding up to a specific sum"
        self.task_description = "Given a list of numbers $l$ separated by spaces and a target value $k$, find two numbers in the list that add up to the target value."
        self.worst_complexity = Complexity.N_SQUARE
        self.best_complexity = Complexity.N
        self.answer_format_requirements = self._answer_format_requirements__list__two_num

    def _prepare_input(self, whatever_input):
        assert isinstance(whatever_input, list)
        assert self._check_example__list(whatever_input)
        if len(whatever_input) == 0:
            raise ValueError(f"[Task.py:TwoSum] ==> Input list cannot be empty: {whatever_input}")

        if len(whatever_input) <= 2:
            raise ValueError(f"[Task.py:TwoSum] ==> Input list must have at least three elements: {whatever_input}")

        value_to_indices: Dict[int, List[int]] = {}
        for index, value in enumerate(whatever_input):
            if value not in value_to_indices:
                value_to_indices[value] = []
            value_to_indices[value].append(index)

        sum_to_pairs: Dict[int, List[Tuple[int, int]]] = {}
        for i in range(len(whatever_input)):
            for j in range(i + 1, len(whatever_input)):
                s = whatever_input[i] + whatever_input[j]
                if s not in sum_to_pairs:
                    sum_to_pairs[s] = []
                sum_to_pairs[s].append((i, j))

        unique_sums = [s for s, pairs in sum_to_pairs.items() if len(pairs) == 1]
        if not unique_sums:
            raise ValueError(f"[Task.py:TwoSum] ==> No unique sum found for input {whatever_input}")

        chosen_sum = random.choice(unique_sums)
        return (whatever_input, chosen_sum)

    def _twosum(self, processed_input: List[int], sum: int) -> List[int]:
        n = len(processed_input)
        for i in range(n):
            for j in range(i + 1, n):
                if processed_input[i] + processed_input[j] == sum:
                    return [processed_input[i], processed_input[j]]
        return []

    def _execute(self, processed_input_tuple: Tuple[List[int], int]):
        input, sum = processed_input_tuple
        numbers = self._twosum(input, sum)
        assert len(numbers) == 2
        return (numbers, numbers[::-1])

    def _check_valid(self, example):
        if not example:
            return False

        if len(example) <= 2:
            return False

        value_to_indices: Dict[int, List[int]] = {}
        for index, value in enumerate(example):
            if value not in value_to_indices:
                value_to_indices[value] = []
            value_to_indices[value].append(index)
        sum_to_pairs: Dict[int, List[Tuple[int, int]]] = {}
        for i in range(len(example)):
            for j in range(i + 1, len(example)):
                s = example[i] + example[j]
                if s not in sum_to_pairs:
                    sum_to_pairs[s] = []
                sum_to_pairs[s].append((i, j))
        unique_sums = [s for s, pairs in sum_to_pairs.items() if len(pairs) == 1]
        if not unique_sums:
            return False
        return True

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 3
            if bool_b:
                return self._check_valid(example)

        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 3

        n_max_trials = 10
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py:TwoSum] ==> This example is invalid: {invalid_example}")

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class ThreeSum(_Task_NumList_Num_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding three numbers adding up to a specific sum"
        self.task_description = "Given a list of numbers $l$ separated by spaces and a target value $k$, find three numbers in the list that add up to the target value."
        self.worst_complexity = Complexity.N_CUBE
        self.best_complexity = Complexity.N_SQUARE
        self.answer_format_requirements = self._answer_format_requirements__list__three_num

    def _prepare_input(self, whatever_input):
        assert isinstance(whatever_input, list)
        assert self._check_example__list(whatever_input)
        if len(whatever_input) == 0:
            raise ValueError(f"[Task.py:ThreeSum] ==> Input list cannot be empty: {whatever_input}")

        if len(whatever_input) <= 3:
            raise ValueError(f"[Task.py:ThreeSum] ==> Input list must have at least four elements: {whatever_input}")

        sum_to_triples: Dict[int, List[Tuple[int, int, int]]] = {}
        for i in range(len(whatever_input)):
            for j in range(i + 1, len(whatever_input)):
                for k in range(j + 1, len(whatever_input)):
                    s = whatever_input[i] + whatever_input[j] + whatever_input[k]
                    if s not in sum_to_triples:
                        sum_to_triples[s] = []
                    sum_to_triples[s].append((i, j, k))

        unique_sums = [s for s, triples in sum_to_triples.items() if len(triples) == 1]
        if not unique_sums:
            raise ValueError("No unique sum found")

        chosen_sum = random.choice(unique_sums)

        return (whatever_input, chosen_sum)

    def _threesum(self, processed_input: List[int], sum: int) -> List[int]:
        n = len(processed_input)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if processed_input[i] + processed_input[j] + processed_input[k] == sum:
                        return [processed_input[i], processed_input[j], processed_input[k]]
        return []

    def _execute(self, processed_input_tuple: Tuple[List[int], int]):
        input, sum = processed_input_tuple
        numbers = self._threesum(input, sum)
        assert len(numbers) == 3
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not example:
            return False

        if len(example) <= 3:
            return False

        sum_to_triples: Dict[int, List[Tuple[int, int, int]]] = {}
        for i in range(len(example)):
            for j in range(i + 1, len(example)):
                for k in range(j + 1, len(example)):
                    s = example[i] + example[j] + example[k]
                    if s not in sum_to_triples:
                        sum_to_triples[s] = []
                    sum_to_triples[s].append((i, j, k))

        unique_sums = [s for s, triples in sum_to_triples.items() if len(triples) == 1]
        if not unique_sums:
            return False
        return True

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 4
            if bool_b:
                return self._check_valid(example)

        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 4

        n_max_trials = 10
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py:ThreeSum] ==> This example is invalid: {invalid_example}")

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class FourSum(_Task_NumList_Num_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding four numbers adding up to a specific sum"
        self.task_description = "Given a list of numbers $l$ separated by spaces and a target value $k$, find four numbers in the list that add up to the target value."
        self.worst_complexity = Complexity.N_POWER_4
        self.best_complexity = Complexity.N_CUBE
        self.answer_format_requirements = self._answer_format_requirements__list__four_num

    def _prepare_input(self, whatever_input: List[int]) -> Tuple[List[int], int]:
        assert isinstance(whatever_input, list)

        if len(whatever_input) == 0:
            raise ValueError(f"[Task.py:FourSum] ==> Input list cannot be empty: {whatever_input}")

        if len(whatever_input) <= 4:
            raise ValueError(f"[Task.py:FourSum] ==> Input list must have at least five elements: {whatever_input}")

        sum_to_quads: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for i in range(len(whatever_input)):
            for j in range(i + 1, len(whatever_input)):
                for k in range(j + 1, len(whatever_input)):
                    for l in range(k + 1, len(whatever_input)):
                        s = whatever_input[i] + whatever_input[j] + whatever_input[k] + whatever_input[l]
                        if s not in sum_to_quads:
                            sum_to_quads[s] = []
                        sum_to_quads[s].append((i, j, k, l))

        unique_sums = [s for s, quads in sum_to_quads.items() if len(quads) == 1]
        if not unique_sums:
            raise ValueError("No unique sum found")

        chosen_sum = random.choice(unique_sums)

        return (whatever_input, chosen_sum)

    def _foursum(self, processed_input: List[int], sum: int) -> List[int]:
        n = len(processed_input)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if processed_input[i] + processed_input[j] + processed_input[k] + processed_input[l] == sum:
                            return [processed_input[i], processed_input[j], processed_input[k], processed_input[l]]
        return []

    def _execute(self, processed_input_tuple: Tuple[List[int], int]):
        input, sum = processed_input_tuple
        numbers = self._foursum(input, sum)
        assert len(numbers) == 4
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not example:
            return False

        if len(example) <= 4:
            return False

        sum_to_triples: Dict[int, List[Tuple[int, int, int]]] = {}
        for i in range(len(example)):
            for j in range(i + 1, len(example)):
                for k in range(j + 1, len(example)):
                    for l in range(k + 1, len(example)):
                        s = example[i] + example[j] + example[k] + example[l]
                        if s not in sum_to_triples:
                            sum_to_triples[s] = []
                        sum_to_triples[s].append((i, j, k, l))

        unique_sums = [s for s, triples in sum_to_triples.items() if len(triples) == 1]
        if not unique_sums:
            return False
        return True

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 5
            if bool_b:
                return self._check_valid(example)

        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 5

        n_max_trials = 50
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py:FourSum] ==> This example is invalid: {invalid_example}")

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class SubsetSum(_Task_NumList_Num_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding the subset of numbers adding up to a specific sum"
        self.task_description = "Given a list of numbers $l$ separated by spaces and a target value $k$, find a set of numbers in the list that add up to the target value."
        self.worst_complexity = Complexity.EXPONENTIAL
        self.best_complexity = Complexity.EXPONENTIAL

        self._min_subset_size = 4
        self._max_subset_size = 6
        self._min_set_size = 6
        self._max_set_size = 16

    def _find_unique_sums(self, example: List[int]):
        n = len(example)
        assert self._min_set_size <= n <= self._max_set_size

        sum_to_subsets: Dict[int, List[List[int]]] = {}

        for mask in range(1, 1 << n):
            subset = [example[i] for i in range(n) if mask & (1 << i)]
            subset_sum = sum(subset)
            if subset_sum not in sum_to_subsets:
                sum_to_subsets[subset_sum] = []
            sum_to_subsets[subset_sum].append(subset)

        unique_sums = [
            s
            for s, subsets in sum_to_subsets.items()
            if len(subsets) == 1
            and len(subsets[0]) >= self._min_subset_size
            and len(subsets[0]) <= self._max_subset_size
        ]

        return unique_sums

    def _prepare_input(self, whatever_input):
        assert isinstance(whatever_input, list)
        assert self._check_example__list(whatever_input)
        if not whatever_input:
            raise ValueError("Input list cannot be empty")

        if len(whatever_input) < 1:
            raise ValueError("Input list must have at least one element")

        unique_sums = self._find_unique_sums(whatever_input)
        chosen_sum = random.choice(unique_sums)

        return (whatever_input, (chosen_sum))

    def _subsetsum(self, processed_input_tuple: Tuple[List[int], int]) -> Optional[List[int]]:
        processed_input, target_sum = processed_input_tuple

        def backtrack(start: int, current_sum: int, path: List[int]) -> Optional[List[int]]:
            if current_sum == target_sum:
                return path
            if current_sum > target_sum or start >= len(processed_input):
                return None

            for i in range(start, len(processed_input)):
                result = backtrack(i + 1, current_sum + processed_input[i], path + [i])
                if result is not None:
                    return result

            return None

        return backtrack(0, 0, [])

    def _execute(self, processed_input_tuple: Tuple[List[int], int]):
        numbers = self._subsetsum(processed_input_tuple)
        assert (
            numbers is not None
            and isinstance(numbers, list)
            and self._min_subset_size <= len(numbers) <= self._max_subset_size
        )
        valid_answers = [[processed_input_tuple[0][i] for i in p] for p in permutations(numbers)]
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not isinstance(example, list):
            return False

        if not example:
            return False

        if len(example) < self._min_set_size:
            return False

        if len(example) > self._max_set_size:
            return False

        unique_sums = self._find_unique_sums(example)

        if unique_sums:
            return True
        else:
            return False

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = self._min_set_size <= len(example) <= self._max_set_size
            if bool_b:
                return self._check_valid(example)

        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert self._min_set_size <= len(invalid_example) <= self._max_set_size

        n_max_trials = 10
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py] ==> This example is invalid: {invalid_example}")

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class LongestCommonSubarray(_Task_NumListPair_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding the longest common subarray"
        self.task_description = "Given two integer arrays $l_1$ and $l_2$, return the longest common subarray that appears in both arrays. A subarray is a contiguous sequence of numbers within an array."
        self.worst_complexity = Complexity.N_CUBE
        self.best_complexity = Complexity.N_LOG_N

    def _prepare_input(self, whatever_input: List[List[int]]) -> Tuple[List[int], List[int]]:
        # Randomly split into two sublists
        if len(whatever_input) != 2:
            raise ValueError("Input must contain exactly two rows")

        row1, row2 = whatever_input
        return (row1, row2)

    def _execute(self, input: Tuple[List[int], List[int]]) -> List[int]:
        nums1, nums2 = input
        m, n = len(nums1), len(nums2)

        # Initialize DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        max_length = 0
        end_index_nums1 = 0

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_index_nums1 = i
                else:
                    dp[i][j] = 0

        # Reconstruct the longest common subarray
        start_index = end_index_nums1 - max_length
        return (nums1[start_index:end_index_nums1],)

    def has_unique_longest_common_subarray(self, input: Tuple[List[int], List[int]]) -> bool:
        nums1, nums2 = input
        m, n = len(nums1), len(nums2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        max_length = 0
        subarrays = set()

        # Fill the DP table and track all longest common subarrays
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        subarrays = {tuple(nums1[i - max_length : i])}
                    elif dp[i][j] == max_length:
                        subarrays.add(tuple(nums1[i - max_length : i]))
                else:
                    dp[i][j] = 0

        # Determine if there is exactly one unique longest common subarray
        return len(subarrays) == 1

    def check_example(self, example):
        if not self._check_example__list_pair(example):
            return False

        example1, example2 = example

        for e in [example1, example2]:
            if len(e) < 2:
                return False

        return self.has_unique_longest_common_subarray(example)

    def _try_transform_invalid_example(self, invalid_example: List[List[int]]) -> List[List[int]]:
        assert self._check_example__list_pair(invalid_example)

        example_list = [list(lst) for lst in invalid_example]

        n_max_trials = 10
        n_trials = 0
        while not self.check_example(example_list):
            list_index = random.choice([0, 1])
            element_index = random.randint(0, len(example_list[list_index]) - 1)
            example_list[list_index][element_index] += 1
            n_trials += 1
            if n_trials == n_max_trials:
                raise ValueError(f"[Task.py:LongestCommonSubarray] ==> This example is invalid: {example_list}")

        return example_list

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        number_range = (1_000, 9_999)

        with open(id_elements_path, "r") as f:
            id_elements = json.load(f)

        unique_id_elements = list(set(id_elements))

        unique_ood_elements = list(set(list(range(number_range[0], number_range[1] + 1))) - set(unique_id_elements))
        assert len(unique_ood_elements) > 0

        ood_elements = np.random.choice(unique_ood_elements, len(id_elements), replace=True).tolist()

        with open(id_examples_path, "r") as f:
            id_examples = json.load(f)

        assert all(isinstance(e, list) for e in id_examples)
        for example in id_examples:
            for row in example:
                assert isinstance(row, list)
                for element in row:
                    assert not isinstance(
                        element, list
                    ), f"Nested list of more than 2 levels is not supported: {example}"

        ood_examples = []
        for id_example in tqdm(id_examples):
            n_max_trials = 10
            n_trials = 0
            while True:
                ood_example = []
                assert len(id_example) == 2
                id_row1, id_row2 = id_example
                ood_row1 = np.random.choice(ood_elements, len(id_row1), replace=True).tolist()
                ood_row2 = []
                for i in range(len(id_row2)):
                    an_element_from_ood_row1 = ood_row1[i] if i < len(ood_row1) else random.choice(ood_elements)
                    an_element_from_ood_elements = random.choice(ood_elements)
                    element_to_append = random.choice([an_element_from_ood_row1, an_element_from_ood_elements])
                    ood_row2.append(element_to_append)
                ood_example.append(ood_row1)
                ood_example.append(ood_row2)

                if self.check_example(ood_example):
                    ood_examples.append(ood_example)
                    break
                else:
                    try:
                        ood_example = self.try_transform_invalid_example(ood_example)
                        ood_examples.append(ood_example)
                        break
                    except:
                        pass

                n_trials += 1
                if n_trials >= n_max_trials:
                    breakpoint()

        assert len(ood_examples) == len(id_examples)

        ood_examples = [format_element_or_example(e, self) for e in ood_examples if self.check_example(e)]
        ood_elements = [get_elements_from_example(e, self) for e in ood_examples]

        print(f"[Task.py:_generate_ood_data__number_list] ==> Saving OOD elements to {ood_elements_save_path}")
        with open(ood_elements_save_path, "w") as f:
            json.dump(ood_elements, f, ensure_ascii=True, cls=NpEncoder)
        print(f"[Task.py:_generate_ood_data__number_list] ==> Saving OOD examples to {ood_examples_save_path}")
        with open(ood_examples_save_path, "w") as f:
            json.dump(ood_examples, f, ensure_ascii=True, cls=NpEncoder)


#! Done
class TSP(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.task_title = "Traveling Salesman Problem (TSP)"
        self.task_description = "Your task is to solve the Traveling Salesman Problem (TSP). Given a list of cities and the distances between each pair of cities, your goal is to find the shortest path that visits every city once and returns to the starting city. The inputs include 1) $n$: the number of cities; 2) $D$: an adjacency matrix of size $n \\times n$ where $D_{ij}$ is the distance between city $i$ and city $j$. The output should be a list of integers representing the order of cities to visit. The cities are indexed from 0 to $n-1$. City 0 is always the starting city."
        self.example_type_for_gpt_extract: str = "list"
        self.worst_complexity = Complexity.EXPONENTIAL
        self.best_complexity = Complexity.EXPONENTIAL
        self.answer_format_requirements = 'The answer should be a list of integers representing the order of cities to visit, and must start with 0 and also end with 0. For example, "The answer is [0, 1, 2, 3, 0]" is in a valid format.'

        self._min_num_cities = 4
        self._max_num_cities = 10

    def random_test_inputs_prompt(self):
        return "Randomly generate some adjacency matrices as test inputs for testing a program written for the task of solving the Traveling Salesman Problem (TSP). The task description is: Given a list of cities and the distances between each pair of cities, find the shortest path that visits every city once and returns to the starting city. Enclose each adjacency matrix by square brackets. Make sure the matrix is symmetric, and each element should be a non-negative integer. For example, [[0, x1, x2, x3], [x1, 0, x4, x5], [x2, x4, 0, x6], [x3, x5, x6, 0]] is a valid adjacency matrix. Do not generate the corresponding output. Do not use or generate any code. Make sure that the adjacency matrices have length larger than 3 but not too large.\n\nNow please generate as many such adjacency matrices as possible:\n\n"

    def _find_shortest_paths(self, adjacency_matrix: list[list[int]]) -> list[int]:
        n = len(adjacency_matrix)
        all_permutations = permutations(range(1, n))
        all_permutations = [[0] + list(perm) + [0] for perm in all_permutations]

        min_distance = float("inf")
        dist2path = {}
        for perm in all_permutations:
            assert len(perm) == n + 1

            current_distance = 0
            for i in range(n):
                dist = adjacency_matrix[perm[i]][perm[i + 1]]
                current_distance += dist

            if current_distance < min_distance:
                min_distance = current_distance

            if current_distance not in dist2path:
                dist2path[current_distance] = [perm]
            else:
                dist2path[current_distance].append(perm)

        return dist2path[min_distance]

    def _execute(self, processed_input: List[List[int]]):
        paths = self._find_shortest_paths(processed_input)
        assert len(paths) == 2
        return tuple(p for p in paths)

    def _cast_input_to_string(self, input: List[List[int]]) -> str:
        return self._cast_matrix_to_string_for_matrixwithsize(input)

    def _cast_output_to_string(self, output: List[int]) -> str:
        return self._cast_list_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__adjacency_matrix(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(100, 999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def check_example(self, example):
        if not self._check_example__adjacency_matrix(example):
            return False

        if not self._min_num_cities <= len(example) <= self._max_num_cities:
            return False

        paths = self._find_shortest_paths(example)
        if len(paths) > 2:  # minimum number of paths is 2
            return False

        return True

    def _try_transform_invalid_example(self, invalid_example):
        return self._try_transform_invalid_example__adjacency_matrix(
            invalid_example, min_dim=self._min_num_cities, max_dim=self._max_num_cities
        )

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__list(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, reverse_ok=True)


# todo
class ShortestPathFromSourceToTarget(_Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_type = TaskType.ARITHMETIC
        self.task_title = "finding the shortest path from a source node to a target node"
        self.task_description = "Given an adjacency matrix $D$ of size $n \\times n$ where $D_{ij}$ represents the weight of the edge from node $i$ to node $j$, a source node $s$ and a target node $t$, output the shortest path from the source node to the target node."
        self.example_type_for_gpt_extract: str = "list"
        self.worst_complexity = Complexity.EXPONENTIAL
        self.best_complexity = Complexity.N_LOG_N
        self.answer_format_requirements = 'The answer should be a list of integers representing the path from the source node $s$ to the target node $t$. For example, if $s=0$ and $t=3$, then "The answer is [0, 1, 2, 3]" is in a valid format.'

        self._min_num_nodes = 4
        self._max_num_nodes = 10

    def random_test_inputs_prompt(self):
        return "Randomly generate some adjacency matrices as test inputs for testing a program written for the task of finding shortest paths from a source node to a target node. The task description is: Given an adjacency matrix, output the shortest path from a source node to a target node. Enclose each adjacency matrix by square brackets. Make sure the matrix is symmetric, and each element should be a non-negative integer. For example, [[0, x1, x2, x3], [x1, 0, x4, x5], [x2, x4, 0, x6], [x3, x5, x6, 0]] is a valid adjacency matrix. Do not generate the corresponding output. Do not use or generate any code. Make sure that the adjacency matrices have length larger than 3 but not too large."

    def _find_shortest_paths(self, adjacency_matrix: List[List[int]]) -> Tuple[List[int], List[int]]:
        num_nodes = len(adjacency_matrix)

        distances = [float("inf")] * num_nodes
        distances[0] = 0

        predecessors = [-1] * num_nodes

        priority_queue = [(0, 0)]

        trail = 0
        max_trials = 50
        while priority_queue:
            trail += 1
            if trail > max_trials:
                raise ValueError("There is problem with the input{adjacency_matrix}")

            current_distance, u = heapq.heappop(priority_queue)

            if current_distance > distances[u]:
                continue

            for v, weight in enumerate(adjacency_matrix[u]):
                if weight > 0:
                    distance = current_distance + weight

                    if distance < distances[v]:
                        distances[v] = distance
                        predecessors[v] = u
                        heapq.heappush(priority_queue, (distance, v))

        return distances, predecessors

    def _reconstruct_path(self, predecessors: List[int], target: int) -> List[int]:
        assert -1 in predecessors
        path = []
        trail = 0
        max_trials = 50
        while target != -1:
            trail += 1
            if trail > max_trials:
                raise ValueError("There is problem with the input {predecessors} and {target}")
            path.append(target)
            target = predecessors[target]
        path.reverse()
        return path

    def _is_unique_path(self, adjacency_matrix: List[List[int]], predecessors: List[int], target: int) -> bool:
        def dfs(node: int) -> int:
            count = 0
            for v, weight in enumerate(adjacency_matrix[node]):
                if weight > 0 and predecessors[v] == node:
                    count += dfs(v)
            return count if node != target else count + 1

        return dfs(0) == 1

    def _prepare_input(self, adjacency_matrix: List[List[int]]):
        assert self._check_example__adjacency_matrix(adjacency_matrix)
        distances, predecessors = self._find_shortest_paths(adjacency_matrix)

        unique_nodes = [
            target
            for target in range(1, len(adjacency_matrix))
            if self._is_unique_path(adjacency_matrix, predecessors, target)
        ]

        return (adjacency_matrix, random.choice(unique_nodes)) if unique_nodes else (None, None)

    def _execute(self, processed_input) -> List[int]:
        target_node, adjacency_matrix = processed_input
        if target_node == None:
            return []  # No unique path found

        distances, predecessors = self._find_shortest_paths(adjacency_matrix)
        path = self._reconstruct_path(predecessors, target_node)
        return (path,)

    def _cast_input_to_string(self, processed_input_tuple: Tuple[List[int], int]) -> str:
        return self._cast_list_to_string_for_strplusint(processed_input_tuple)

    def _cast_output_to_string(self, output: List[int]) -> str:
        return self._cast_list_to_string(output)

    def _generate_ood_data(self, id_elements_path, id_examples_path, ood_elements_save_path, ood_examples_save_path):
        self._generate_ood_data__adjacency_matrix(
            id_elements_path=id_elements_path,
            id_examples_path=id_examples_path,
            ood_elements_save_path=ood_elements_save_path,
            ood_examples_save_path=ood_examples_save_path,
            number_range=(0, 999),
        )

    def check_element(self, element):
        return self._check_element__non_neg_int(element)

    def check_example(self, example):
        if not self._check_example__adjacency_matrix(example):
            return False

        if not self._min_num_nodes <= len(example) <= self._max_num_nodes:
            return False

        distances, predecessors = self._find_shortest_paths(example)

        unique_nodes = [
            target for target in range(1, len(example)) if self._is_unique_path(example, predecessors, target)
        ]

        if unique_nodes:
            return True
        else:
            return False

    def format_element(self, element):
        return self._format_element__int(element)

    def extract_answer(self, model_output: str) -> str:
        return self._extract_answer__list(model_output)

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, reverse_ok=True)

    def _try_transform_invalid_example(self, invalid_example: List[List[int]]) -> List[List[int]]:
        invalid_example = self._try_transform_invalid_example__adjacency_matrix(
            invalid_example, min_dim=self._min_num_nodes, max_dim=self._max_num_nodes
        )

        assert self._min_num_nodes <= len(invalid_example) <= self._max_num_nodes

        matrix = deepcopy(invalid_example)
        num_nodes = len(matrix)

        def prepare_input(matrix: List[List[int]]) -> int:
            distances, predecessors = self._find_shortest_paths(matrix)
            unique_nodes = [
                target for target in range(1, num_nodes) if self._is_unique_path(matrix, predecessors, target)
            ]
            return random.choice(unique_nodes) if unique_nodes else -1

        n_trials = 0
        n_max_trials = 50
        while True:
            valid_node = prepare_input(matrix)

            if valid_node != -1:
                return matrix

            row = random.randint(0, num_nodes - 1)
            col = random.randint(0, num_nodes - 1)

            while row == col or matrix[row][col] == 0:
                row = random.randint(0, num_nodes - 1)
                col = random.randint(0, num_nodes - 1)

            matrix[row][col] += 1
            matrix[col][row] += 1

            n_trials += 1
            if n_trials == n_max_trials:
                raise ValueError("Cannot transform the invalid example")


#! Done
class ThreeSumInRange(_Task_NumList_NumPair_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding three numbers adding up to be in a specific range"
        self.task_description = "Given a list of numbers $l$ separated by spaces and two numbers $a$ and $b$, find three numbers in the list that add up to a value that is in the range $(a, b)$, i.e. greater than $a$ and less than $b$."
        self.worst_complexity = Complexity.N_CUBE
        self.best_complexity = Complexity.N_SQUARE
        self.answer_format_requirements = self._answer_format_requirements__list__three_num

    def _prepare_input(self, whatever_input: List[int]) -> Tuple[List[int], Optional[Tuple[int, int, int]]]:
        assert isinstance(whatever_input, list)
        assert self._check_example__list(whatever_input)
        sorted_input = sorted(whatever_input)

        sum_triplets = []

        for i in range(len(sorted_input)):
            for j in range(i + 1, len(sorted_input)):
                for k in range(j + 1, len(sorted_input)):
                    triplet_sum = sorted_input[i] + sorted_input[j] + sorted_input[k]
                    sum_triplets.append(triplet_sum)

        sum_triplets = sorted(sum_triplets)

        while True:
            m = random.randint(1, len(sum_triplets) - 2)
            if (sum_triplets[m + 1] != sum_triplets[m]) and (sum_triplets[m - 1] != sum_triplets[m]):
                return (whatever_input, (sum_triplets[m - 1], sum_triplets[m + 1]))

    def _threesum(self, processed_input: List[int], first: int, second: int) -> List[int]:
        n = len(processed_input)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (processed_input[i] + processed_input[j] + processed_input[k] > first) and (
                        processed_input[i] + processed_input[j] + processed_input[k] < second
                    ):
                        return [processed_input[i], processed_input[j], processed_input[k]]
        return []

    def _execute(self, processed_input_tuple):
        input, all = processed_input_tuple
        first, second = all
        numbers = self._threesum(input, first, second)
        assert len(numbers) == 3
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not example:
            return False

        if len(example) <= 3:
            return False
        sorted_input = sorted(example)

        sum_triplets = []
        for i in range(len(sorted_input)):
            for j in range(i + 1, len(sorted_input)):
                for k in range(j + 1, len(sorted_input)):
                    triplet_sum = sorted_input[i] + sorted_input[j] + sorted_input[k]
                    sum_triplets.append(triplet_sum)

        sum_triplets.sort()

        for i in range(1, len(sum_triplets) - 1):
            if (sum_triplets[i + 1] != sum_triplets[i]) and (sum_triplets[i - 1] != sum_triplets[i]):
                return True

        return False

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 4
            if bool_b:
                return self._check_valid(example)
        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 4
        n_max_trials = 50
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py:ThreeSum] ==> This example is invalid: {invalid_example}")

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class FourSumInRange(_Task_NumList_NumPair_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding four numbers adding up to be in a specific range"
        self.task_description = "Given a list of numbers $l$ separated by spaces and two numbers $a$ and $b$, find four numbers in the list that add up to a value that is in the range $(a, b)$, i.e. greater than $a$ and less than $b$."
        self.worst_complexity = Complexity.N_POWER_4
        self.best_complexity = Complexity.N_CUBE
        self.answer_format_requirements = self._answer_format_requirements__list__four_num

    def _prepare_input(self, whatever_input: List[int]) -> Tuple[List[int], Optional[Tuple[int, int, int]]]:
        assert isinstance(whatever_input, list)
        assert self._check_example__list(whatever_input)
        sorted_input = sorted(whatever_input)

        sum_triplets = []

        for i in range(len(sorted_input)):
            for j in range(i + 1, len(sorted_input)):
                for k in range(j + 1, len(sorted_input)):
                    for l in range(k + 1, len(sorted_input)):
                        sum = sorted_input[i] + sorted_input[j] + sorted_input[k] + sorted_input[l]
                        sum_triplets.append(sum)

        sum_triplets = sorted(set(sum_triplets))

        while True:
            m = random.randint(1, len(sum_triplets) - 2)
            if (sum_triplets[m + 1] != sum_triplets[m]) and (sum_triplets[m - 1] != sum_triplets[m]):
                return (whatever_input, (sum_triplets[m - 1], sum_triplets[m + 1]))

    def _foursum(self, processed_input: List[int], first: int, second: int) -> List[int]:
        n = len(processed_input)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if (
                            processed_input[i] + processed_input[j] + processed_input[k] + processed_input[l] > first
                        ) and (
                            processed_input[i] + processed_input[j] + processed_input[k] + processed_input[l] < second
                        ):
                            return [processed_input[i], processed_input[j], processed_input[k], processed_input[l]]
        return []

    def _execute(self, processed_input_tuple):
        input, all = processed_input_tuple
        first, second = all
        numbers = self._foursum(input, first, second)
        assert len(numbers) == 4
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not example:
            return False

        if len(example) <= 4:
            return False
        sorted_input = sorted(example)

        sum_triplets = []
        for i in range(len(sorted_input)):
            for j in range(i + 1, len(sorted_input)):
                for k in range(j + 1, len(sorted_input)):
                    for m in range(k + 1, len(sorted_input)):
                        triplet_sum = sorted_input[i] + sorted_input[j] + sorted_input[k] + sorted_input[m]
                        sum_triplets.append(triplet_sum)

        sum_triplets.sort()

        for i in range(1, len(sum_triplets) - 1):
            if (sum_triplets[i + 1] != sum_triplets[i]) and (sum_triplets[i - 1] != sum_triplets[i]):
                return True

        return False

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 5
            if bool_b:
                return self._check_valid(example)
        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 5

        n_max_trials = 20
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py:FourSum] ==> This example is invalid: {invalid_example}")

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class SubsetSumInRange(_Task_NumList_NumPair_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding a subset adding up to be in a specific range"
        self.task_description = "Given a list of numbers $l$ separated by spaces and two numbers $a$ and $b$, find a subset in the list that adds up to a value that is in the range $(a, b)$, i.e. greater than $a$ and less than $b$."
        self.worst_complexity = Complexity.EXPONENTIAL
        self.best_complexity = Complexity.EXPONENTIAL

        self._min_subset_size = 4
        self._max_subset_size = 6
        self._min_set_size = 6
        self._max_set_size = 16

    def _prepare_input(self, whatever_input: List[int]) -> Tuple[List[int], Optional[Tuple[int, int, int]]]:
        assert isinstance(whatever_input, list)
        assert self._check_example__list(whatever_input)
        sorted_input = sorted(whatever_input)

        sum_subsets = []

        for r in range(self._min_subset_size, self._max_subset_size + 1):
            for subset in combinations(sorted_input, r):
                subset_sum = sum(subset)
                sum_subsets.append(subset_sum)
                sum_subsets = sorted(sum_subsets)

        while True:
            m = random.randint(1, len(sum_subsets) - 2)
            if (sum_subsets[m + 1] != sum_subsets[m]) and (sum_subsets[m - 1] != sum_subsets[m]):
                return (whatever_input, (sum_subsets[m - 1], sum_subsets[m + 1]))

    def _subsetsum(self, processed_input: List[int], first: int, second: int) -> List[int]:
        sorted_input = sorted(processed_input)
        for r in range(self._min_subset_size, self._max_subset_size + 1):
            for subset in combinations(sorted_input, r):
                subset_sum = sum(subset)
                if (subset_sum > first) and (subset_sum < second):
                    return list(subset)
        return []

    def _execute(self, processed_input_tuple):
        input, all = processed_input_tuple
        first, second = all
        numbers = self._subsetsum(input, first, second)
        assert (
            numbers is not None
            and isinstance(numbers, list)
            and self._min_subset_size <= len(numbers) <= self._max_subset_size
        )
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not isinstance(example, list):
            return False

        if not example:
            return False

        if len(example) < self._min_set_size:
            return False

        if len(example) > self._max_set_size:
            return False

        sorted_input = sorted(example)
        sum_subsets = []
        for r in range(self._min_subset_size, self._max_subset_size + 1):
            for subset in combinations(sorted_input, r):
                subset_sum = sum(subset)
                sum_subsets.append(subset_sum)
                sum_subsets = sorted(sum_subsets)

        sum_subsets.sort()

        for i in range(1, len(sum_subsets) - 1):
            if (sum_subsets[i + 1] != sum_subsets[i]) and (sum_subsets[i - 1] != sum_subsets[i]):
                return True

        return False

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = self._min_set_size <= len(example) <= self._max_set_size
            if bool_b:
                return self._check_valid(example)
        return False

    def _cast_input_to_string(self, input) -> str:
        return self._cast_list_to_string_and_tuple(input)

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)

        n_max_trials = 100
        n = 0
        while not self._check_valid(invalid_example):
            n += 1
            index = random.randint(1, len(invalid_example))
            invalid_example[index] += 1
            if n == n_max_trials:
                raise ValueError(f"[Task.py:SubsetSum] ==> This example is invalid: {invalid_example}")

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class ThreeSumMultipleTen(_Task_NumList_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding three numbers adding up to be multiple of 10"
        self.task_description = "Given a list of numbers separated by spaces, find three numbers in the list that add up to be a multiple of 10."
        self.worst_complexity = Complexity.N_CUBE
        self.best_complexity = Complexity.N_SQUARE
        self.answer_format_requirements = self._answer_format_requirements__list__three_num

    def _threesum(self, processed_input: List[int]) -> List[int]:
        n = len(processed_input)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (processed_input[i] + processed_input[j] + processed_input[k]) % 10 == 0:
                        return [processed_input[i], processed_input[j], processed_input[k]]
        return []

    def _execute(self, processed_input):
        numbers = self._threesum(processed_input)
        assert len(numbers) == 3
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not example:
            return False

        if len(example) <= 3:
            return False
        sorted_input = sorted(example)
        n = 0

        for i in range(len(sorted_input)):
            for j in range(i + 1, len(sorted_input)):
                for k in range(j + 1, len(sorted_input)):
                    if (example[i] + example[j] + example[k]) % 10 == 0:
                        n += 1

        return n == 1

    def none_or_more(self, processed_input):
        n = 0
        for i in range(len(processed_input)):
            for j in range(i + 1, len(processed_input)):
                for k in range(j + 1, len(processed_input)):
                    if (processed_input[i] + processed_input[j] + processed_input[k]) % 10 == 0:
                        n += 1
        return n == 0

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            return self._check_valid(example)
        return False

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 4
        n_max_trials = 50
        n_stop = 10
        n = 0
        flag = 0
        while not self._check_valid(invalid_example):
            flag += 1
            if flag == n_max_trials:
                raise ValueError(f"[Task.py:Fournumber] ==> This example is invalid: {invalid_example}")
            if self.none_or_more(invalid_example):

                while self.none_or_more(invalid_example):
                    n += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py:Fournumber] ==> This example is invalid: {invalid_example}")
                    random_index = random.randint(0, len(invalid_example) - 1)
                    invalid_example[random_index] += 1
            else:
                random_number = 0
                n = 0
                while random_number % 10 == 0:
                    random_number = random.choice(invalid_example)
                    n += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py:Fournumber] ==> This example is invalid: {invalid_example}")
                    if n == n_max_trials:
                        random_number += 1
                        break
                n = 0
                while not self._check_valid(invalid_example):
                    n += 1
                    if n % n_stop == 0:
                        random_number += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py] ==> This example is invalid: {invalid_example}")
                    random_index = random.randint(0, len(invalid_example) - 1)
                    invalid_example[random_index] = random_number

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class FourSumMultipleTen(_Task_NumList_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding four numbers adding up to be multiple of 10"
        self.task_description = "Given a list of numbers separated by spaces, find four numbers in the list that add up to be a multiple of 10."
        self.worst_complexity = Complexity.N_POWER_4
        self.best_complexity = Complexity.N_CUBE
        self.answer_format_requirements = self._answer_format_requirements__list__four_num

    def _foursum(self, processed_input: List[int]) -> List[int]:
        n = len(processed_input)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if (
                            processed_input[i] + processed_input[j] + processed_input[k] + processed_input[l]
                        ) % 10 == 0:
                            return [processed_input[i], processed_input[j], processed_input[k], processed_input[l]]
        return []

    def _execute(self, processed_input):
        numbers = self._foursum(processed_input)
        assert len(numbers) == 4
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, example):
        if not example:
            return False

        if len(example) <= 4:
            return False
        n = 0
        sorted_input = sorted(example)

        for i in range(len(sorted_input)):
            for j in range(i + 1, len(sorted_input)):
                for k in range(j + 1, len(sorted_input)):
                    for m in range(k + 1, len(sorted_input)):
                        triplet_sum = sorted_input[i] + sorted_input[j] + sorted_input[k] + sorted_input[m]
                        if triplet_sum % 10 == 0:
                            n += 1
        return n == 1

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = len(example) >= 5
            if bool_b:
                return self._check_valid(example)
        return False

    def _cast_input_to_string(self, input) -> str:
        return self._cast_list_to_string(input)

    def none_or_more(self, processed_input):
        n = 0
        for i in range(len(processed_input)):
            for j in range(i + 1, len(processed_input)):
                for k in range(j + 1, len(processed_input)):
                    for l in range(k + 1, len(processed_input)):
                        if (
                            processed_input[i] + processed_input[j] + processed_input[k] + processed_input[l]
                        ) % 10 == 0:
                            n += 1
        return n == 0

    def _try_transform_invalid_example(self, invalid_example):
        assert self._check_example__list(invalid_example)
        assert len(invalid_example) >= 5
        n_max_trials = 50
        n_stop = 10
        n = 0
        flag = 0
        while not self._check_valid(invalid_example):
            flag += 1
            if flag == n_max_trials:
                raise ValueError(f"[Task.py:Fournumber] ==> This example is invalid: {invalid_example}")
            if self.none_or_more(invalid_example):
                while self.none_or_more(invalid_example):
                    n += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py:Fournumber] ==> This example is invalid: {invalid_example}")
                    random_index = random.randint(0, len(invalid_example) - 1)
                    invalid_example[random_index] += 1
            else:
                random_number = 0
                n = 0
                while random_number % 10 == 0:
                    random_number = random.choice(invalid_example)
                    n += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py:Fournumber] ==> This example is invalid: {invalid_example}")

                    if n == n_max_trials:
                        random_number += 1
                        break
                n = 0
                while not self._check_valid(invalid_example):
                    n += 1
                    if n % n_stop == 0:
                        random_number += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py] ==> This example is invalid: {invalid_example}")
                    random_index = random.randint(0, len(invalid_example) - 1)
                    invalid_example[random_index] = random_number

        return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


#! Done
class SubsetSumMultipleTen(_Task_NumList_2_NumList):
    def __init__(self) -> None:
        super().__init__()

        self.task_title = "finding a subset adding up to be multiple of 10"
        self.task_description = "Given a list of numbers separated by spaces, find a subset in the list that adds up to be a multiple of 10."
        self.worst_complexity = Complexity.EXPONENTIAL
        self.best_complexity = Complexity.EXPONENTIAL

        self._min_subset_size = 4
        self._max_subset_size = 6
        self._min_set_size = 6
        self._max_set_size = 16

    def _subsetsum(self, processed_input: List[int]) -> List[int]:
        sorted_input = sorted(processed_input)
        for r in range(self._min_subset_size, self._max_subset_size + 1):
            for subset in combinations(sorted_input, r):
                subset_sum = sum(subset)
                if subset_sum % 10 == 0:
                    return list(subset)
        return []

    def _execute(self, input):
        numbers = self._subsetsum(input)
        assert self._min_subset_size <= len(numbers) <= self._max_subset_size
        valid_answers = list(permutations(numbers))
        return tuple(list(ans) for ans in valid_answers)

    def _check_valid(self, processed_input):
        if not isinstance(processed_input, list):
            return False

        if not processed_input:
            return False
        if not isinstance(processed_input, list):
            return False

        if not processed_input:
            return False

        if len(processed_input) < self._min_set_size:
            return False

        if len(processed_input) > self._max_set_size:
            return False

        flag = 0
        sorted_input = sorted(processed_input)
        for r in range(self._min_subset_size, self._max_subset_size + 1):
            for subset in combinations(sorted_input, r):
                subset_sum = sum(subset)
                if subset_sum % 10 == 0:
                    flag += 1
        return flag == 1

    def check_example(self, example):
        bool_a = self._check_example__list(example)
        if bool_a:
            bool_b = self._min_set_size <= len(example) <= self._max_set_size
            if bool_b:
                return self._check_valid(example)
        return False

    def _cast_input_to_string(self, input) -> str:
        return self._cast_list_to_string(input)

    def none_or_more(self, processed_input):
        flag = 0
        sorted_input = sorted(processed_input)
        for r in range(self._min_subset_size, self._max_subset_size + 1):
            for subset in combinations(sorted_input, r):
                subset_sum = sum(subset)
                if subset_sum % 10 == 0:
                    flag += 1
        return flag == 0

    def _try_transform_invalid_example(self, invalid_example):
        n_max_trials = 100
        n_stop = 10
        flag = 0
        while not self._check_valid(invalid_example):
            flag += 1
            if flag == n_max_trials:
                raise ValueError(f"[Task.py:Subsetnumber] ==> This example is invalid: {invalid_example}")

            if self.none_or_more(invalid_example):
                n = 0

                while self.none_or_more(invalid_example):
                    n += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py:Subsetnumber] ==> This example is invalid: {invalid_example}")
                    random_index = random.randint(0, len(invalid_example) - 1)
                    invalid_example[random_index] += 1
            else:
                random_number = 0
                n = 0
                while random_number % 10 == 0:
                    random_number = random.choice(invalid_example)

                    n += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py:Subsetnumber] ==> This example is invalid: {invalid_example}")
                    if n == n_max_trials:
                        random_number += 1
                        break
                n = 0
                while not self._check_valid(invalid_example):
                    n += 1
                    if n % n_stop == 0:
                        random_number += 1
                    if n == n_max_trials:
                        raise ValueError(f"[Task.py:Subsetnumber] ==> This example is invalid: {invalid_example}")
                    random_index = random.randint(0, len(invalid_example) - 1)
                    invalid_example[random_index] = random_number
            return invalid_example

    def evaluate(self, model_answer_js: Optional[str], desired_output_js: str) -> bool:
        return self._evaluate__number_list(model_answer_js, desired_output_js, unordered_ok=True)


task2level = {
    "FindMinimum": "O(N)",
    "FindMaximum": "O(N)",
    "FindMode": "O(N)",
    "FindTopk": "O(N-N^2)",
    "SortNumbers": "O(N-N^2)",
    "RemoveDuplicateNumbers": "O(N-N^2)",
    "TwoSum": "O(N-N^2)",
    "ThreeSum": "O(N^2-N^3)",
    "FourSum": "O(N^3-N^4)",
    "SubsetSum": "O(2^N)",
    "TSP": "O(2^N)",
    "ThreeSumInRange": "O(N^2-N^3)",
    "FourSumInRange": "O(N^3-N^4)",
    "SubsetSumInRange": "O(2^N)",
    "ThreeSumMultipleTen": "O(N^2-N^3)",
    "FourSumMultipleTen": "O(N^3-N^4)",
    "SubsetSumMultipleTen": "O(2^N)"
}
