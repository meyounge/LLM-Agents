import random

def sort_list(data):
    return sorted(data)

# Example usage:
unsorted_list = [random.randint(1, 100) for _ in range(10)]
sorted_list = sort_list(unsorted_list)
print(f"Unsorted: {unsorted_list}")
print(f"Sorted: {sorted_list}")
