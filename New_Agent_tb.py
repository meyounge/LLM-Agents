# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:28:08 2024

@author: Michael Younger
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import ArtificialAgents as AA

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
    Note: Shor's algorithm is meant for factoring numbers so if the input is 35 the output should be 5 or 7, do not be immedietly discouraged if it is not as this algorithm can fail, so be sure to run it a few times to be certain that it does not work
"""

# AA.API_KEYS['chatGPT'] = ''
AA.API_KEYS['Local'] = 'http://localhost:8080'

# model = AA.chatGPT(model='gpt-4', api_key=AA.API_KEYS['chatGPT'])
model = AA.Local(url=AA.API_KEYS['Local'])

# agent = AA.Agent(model=model, functions=AA.default_function_list)

# print(agent.taskallocate(task1, verbose=True))


task2 = """
Code a machine learning algorithm that can classify digits the data set information is given in the file Dataset_information.txt in the home directory (./)
The dataset directory you will be using is ./Train_Test_Data you may list existing files in said directory if you don't believe me. However, be sure to add
os.chdir(os.path.dirname(os.path.abspath(__file__))) to your code, otherwise it may not be using the home directory to run it

Here is how you should go about completing this task (ALWAYS BE SURE THAT EVERY AGENT KNOWS WHAT ANY OTHER DEDICATED AGENT HAS ALREADY DONE AND WHAT FILES ARE AVAILIABLE (you could just tell them something like check the existing files for existing completion) (you do not have to tell them what functions they have availible, they are aware)):
    Step 0: Create a file that will house this project, give it a good name like digit_classifier.py
    Step 1: Dedicate an agent to create a python file that can import the data and be sure that it can; The agent may not tell you how it saved the file, if this is the case, you can utilize tools like 'List Files in Directory' to be able to view what has changed or where the main file is stored
    Step 2: Dedicate an agent to coding a simple CNN machine learning algorithm using pytorch in the same file as before (Only pytorch is installed, we do have a GPU you can use as well so cuda is availiable for training)
    Step 3: Dedicate an agent to coding a training loop for the process, let them decide on the alpha, the batch size, and the loss function to use
    Step 4: Have an agent run tests on the code and append print statements where needed to get better insight into wheather or not it is working, if it is not dedicate another agent to diagnose the problem and refactor and rewrite the existing code
        Note: Be clear about what tests you want to run, first you should make sure it compiles, then you should write print statements for accuracy, and anything else you think may be necessary, also your designated agent for troubleshooting should probably know what the data we're using is (so should all of the agents you create to be fair) and so you should be sure to mention where that information is stored (./Dataset_information.txt)
    Step 5: Be sure the output is as you want it to be, this algorithm does not have to be amazing, it simply has to work well enough to get a passing grade, lets say 70% right on the testing data is enough
    Step 6: If you've met all of the above requirements then you are done
"""

task3 = """
Code a machine learning that utilizes a pytorch CNN machine learning algorithm, that can classify digits the data set information is given in the file Dataset_information.txt in the home directory (./)
The dataset directory you will be using is ./Train_Test_Data you may list existing files in said directory. Be sure to add os.chdir(os.path.dirname(os.path.abspath(__file__))) to your code, otherwise it may not be using the home directory to run it
"""

leader = AA.LeadAgent(model=model, functions=AA.default_function_list)
print(leader.taskallocate(task3, verbose=True))