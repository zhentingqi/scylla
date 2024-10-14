import re
import json
import random
import numpy as np


def extract_number_list_after_keyword(input_string, keywords=["answer is", "answer:", "answer is:", "answer to the task is", "numbers are", "numbers are:", "numbers:"]) -> str:
    """IMPORTANT: The function assumes that the JSON list is enclosed in square brackets []."""
    keywords.append("") # Last shot: Try extracting the last occurrence of a JSON list
    
    for keyword in keywords:
        # Regex pattern to find the JSON list object after "answer is "
        pattern = rf"{keyword}" + r"\s*(\[.*?\])"
        
        # Search for the pattern in the input string
        matches = re.findall(pattern, input_string)

        if matches:
            # Extract the JSON list string
            json_list_string = matches[-1]

            try:
                # Convert the JSON string to a Python list
                _ = json.loads(json_list_string)
                return json_list_string
            except json.JSONDecodeError:
                # print(f"The extracted string: {json_list_string} is not a valid JSON list.")
                pass

    return None
    
    
def extract_number_after_keyword(input_string, keywords=["answer is", "answer:", "answer is:", "answer to the task is"]) -> str:
    """IMPORTANT: The function assumes that the number is a float or integer."""
    keywords.append("") # Last shot: Try extracting the last occurrence of a number
    
    for keyword in keywords:
        # Regex pattern to find the number (float or integer) after "answer is "
        pattern = rf"{keyword}" + r"\s*([0-9]+\.?[0-9]*)"
        
        # Search for the pattern in the input string
        matches = re.findall(pattern, input_string)
        
        if matches:
            # Extract the number string
            number_string = matches[-1]

            try:
                # Convert the number string to a Python float or integer
                number_string = json.dumps(float(number_string))
                _ = json.loads(number_string)
                return number_string
            except ValueError or json.JSONDecodeError:
                # print(f"The extracted string: {number_string} is not a valid number.")
                pass

    return None
    
    
def extract_string_after_keyword(input_string, keywords=["answer is", "answer:", "answer is:", "answer to the task is"]) -> str:
    for keyword in keywords:
        # Regex pattern to find the string after "answer is "
        pattern = rf"{keyword}" + r'\s*(.+)\.'

        # Search for the pattern in the input string
        matches = re.findall(pattern, input_string)

        if matches:
            # Extract the string
            extracted_string = matches[-1]
            
            try:
                # Check if the extracted string is a valid JSON string
                extracted_string = json.dumps(extracted_string, ensure_ascii=False)
                _ = json.loads(extracted_string)
                return extracted_string
            except json.JSONDecodeError:
                # print(f"The extracted string: {extracted_string} is not a valid JSON string.")
                pass

    return None
    
    
def generate_adjacency_matrix(n, pool):
    # Initialize an n x n matrix with zeros
    matrix = np.zeros((n, n), dtype=int)

    # Iterate over the upper triangle of the matrix
    for i in range(n):
        for j in range(i + 1, n):
            # Randomly select a number from the pool
            number = np.random.choice(pool)
            # Assign the number to both [i, j] and [j, i] to ensure symmetry
            matrix[i, j] = number
            matrix[j, i] = number

    matrix = matrix.tolist()
    
    # Make every element an `int` type
    matrix = [[int(e) for e in row] for row in matrix]
    return matrix
    
    
def generate_non_ascii_string(n, unicode_ranges=None):
    if unicode_ranges is None:
        # Define default Unicode ranges (excluding ASCII range)
        unicode_ranges = [
            (0x0370, 0x03FF),  # Greek
            (0x0400, 0x04FF),  # Cyrillic
            (0x0530, 0x058F),  # Armenian
            (0x0590, 0x05FF),  # Hebrew
            (0x0600, 0x06FF),  # Arabic
            (0x0900, 0x097F),  # Devanagari
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs (Chinese characters)
        ]

    # Generate a random non-ASCII string of length n
    non_ascii_string = ''.join(
        chr(random.randint(start, end))
        for _ in range(n)
        for start, end in [random.choice(unicode_ranges)]
    )
    
    return non_ascii_string


def mutate_ood_string(s, num_chars_to_mutate=1, unicode_ranges=None):
    # Mutate: add/remove/replace a random character in the string
    s = list(s)
    assert len(s) > 1, "The input string must not be empty."
    for _ in range(num_chars_to_mutate):
        # Randomly select a character to mutate
        i = random.randint(0, len(s) - 1)
        # Randomly select a mutation operation
        mutation_op = random.choice(["add", "remove", "replace"])
        if mutation_op == "add":
            # Add a random non-ASCII character
            s.insert(i, generate_non_ascii_string(1, unicode_ranges))
        elif mutation_op == "remove":
            # Remove the character at index i
            s.pop(i)
        elif mutation_op == "replace":
            # Replace the character at index i with a random non-ASCII character
            s[i] = generate_non_ascii_string(1, unicode_ranges)
    return ''.join(s)


def get_elements_from_example(obj, task):
    elements = []
    def _add_element_from_example(obj):
        if task.check_element(obj):
            elements.append(obj)
            return
        elif isinstance(obj, list):
            for o in obj:
                _add_element_from_example(o)
        elif isinstance(obj, str) and len(obj) > 1:
            for o in obj:
                _add_element_from_example(o)
        else:
            return
    _add_element_from_example(obj)
    elements = [task.format_element(e) for e in elements]
    return elements


def format_element_or_example(obj, task):
    if task.check_element(obj):
        return task.format_element(obj)
    elif isinstance(obj, list):
        return [format_element_or_example(o, task) for o in obj]
    elif isinstance(obj, str) and len(obj) > 1:
        return "".join([format_element_or_example(o, task) for o in obj])
    else:
        return obj


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
    
if __name__ == '__main__':
    print()
    