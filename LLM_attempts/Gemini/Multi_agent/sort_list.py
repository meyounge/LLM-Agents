# Function to sort a list of numbers in ascending order and handle errors
def sort_numbers(input_list):
    """Sorts a list of numbers in ascending order and handles errors.

    Args:
        input_list: The list of numbers to sort.

    Returns:
        A new list containing the sorted numbers, or an error message if the input is invalid.
    """
    if not isinstance(input_list, list):
        return "Error: Input must be a list."

    numeric_list = []
    for item in input_list:
        try:
            numeric_list.append(float(item))
        except ValueError:
            return "Error: List contains non-numeric values."

    numeric_list.sort()
    return numeric_list

# Get input from the user
# In a real application, you would likely get input from a file or command line arguments.
input_list = [1, 5, 2, 8, 3]

# Sort the list
sorted_list = sort_numbers(input_list)

# Print the sorted list or error message
print(sorted_list)
