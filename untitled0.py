# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:55:08 2024

@author: micha
"""

import os
working_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(working_directory)
print("Working directory:\n" + os.getcwd())

import ArtificialAgents as AA

model = AA.Local(url='http://localhost:8080')

agent = AA.Agent(model = model, functions=AA.default_function_list)

task0 = "Sort a list or array in python"

task1 = """
Code a classical implementation of shor's algorithm using the following steps (N is the input number to the function that is to be factored):
    Step 0: Ensure N is positive and non-trivial and if it is not return 1
    Step 1: Ensure N is not an even number and if it is return 2
    Step 2: Ensure that there is no 'a' such that a**b==N and if there is return it
    step 3: Choose a random positive non-trivial integer 'x' that is less than N and check once without looping that it does not share a common factor with N and if it does return it
    Step 4: Find r such that x**r % N == 1
    Step 5: If r is even we've found a number with a common factor to N defined as x**r / 2 + 1 xor x**r / 2 - 1, try both and return a non-trivial greatest common denominator (gcd)
    Step 6: If none of the above worked we may have failed or found a prime number, we don't know for certain, return N
    Step 7: Outside of the function write a test case for the function make sure to print the output
    Step 8: Run the file and verify it works as intended
    Note: Shor's algorithm is meant for factoring numbers so if the input is 35 the output should be 5 or 7, do not be immediately discouraged if it is not as this algorithm can fail, so be sure to run it a few times to be certain that it does not work
"""

task2 = """
Code a Machine Learning that utilizes a pytorch CNN machine learning algorithm, that can classify digits the data set information is given in the file Dataset_information.txt in the home directory (./)
The dataset directory you will be using is ./Train_Test_Data you may list existing files in said directory. Be sure to add os.chdir(os.path.dirname(os.path.abspath(__file__))) to your code, otherwise it may not be using the home directory to run it
"""

print(agent.taskallocate(task2, verbose=True))