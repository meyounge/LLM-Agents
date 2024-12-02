# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:46:29 2024

@author: micha
"""

my_list = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_list = sorted(my_list)
print(sorted_list)  # Output: [1, 1, 2, 3, 4, 5, 6, 9]
print(my_list)     # Output: [3, 1, 4, 1, 5, 9, 2, 6] (original list unchanged)

sorted_list_desc = sorted(my_list, reverse=True)
print(sorted_list_desc) # Output: [9, 6, 5, 4, 3, 2, 1, 1]