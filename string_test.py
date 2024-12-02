# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:46:22 2024

@author: Michael Younger
"""

prompt = """
You are a friendly AI assistant designed to provide helpful, succinct, and accurate information.

Answer the following questions and obey the following commands as best you can.

You have access to the following tools:

Search: Search: useful for when you need to answer questions about current events. You should ask targeted questions.
Calculator: Useful for when you need to answer questions about math. Use python code, eg: 2 + 2
Response To Human: When you need to respond to the human you are talking to.

You will receive a message from the human, then you should start a loop and do one of two things

Option 1: You use a tool to answer the question.
For this, you should use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: "the input to the action, to be sent to the tool"

After this, the human will respond with an observation, and you will continue.

Option 2: You respond to the human.
For this, you should use the following format:
Action: Response To Human
Action Input: "your response to the human, summarizing what you did and what you learned"

Begin!

User: Who is set to star in the new Your Lie In April musical?

Assistant: Thought: The user is asking about a new musical, and I need to find the cast list.
Action: Search
Action Input: "Your Lie In April musical cast list"

Please respond with the first part of the loop. 

"""

import re

def denoise(input_string):
    # Remove everything before the last "Action Input"
    lines = input_string.splitlines()
    print(lines)
    
    processed_list = []
    for i, string in enumerate(lines):
        if "Action" in string:
            processed_list.append(string)
            
    if len(processed_list) < 2:
        return ''
   
    return processed_list[0] + '\n' + processed_list[1]

# Example usage
test_string = """
Some text here
User: Keep this
Action: search
Action Input: "hello world"
Delete this text after newline
"""

result = denoise(test_string)
print(bool(result))




