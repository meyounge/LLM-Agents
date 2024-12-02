# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:13:07 2024

@author: Michael Younger
"""

import requests
import os, re
from googleapiclient.discovery import build
os.chdir(os.path.dirname(os.path.abspath(__file__)))

default_parameters = {'temperature': 0.8, 'top_k': 35, 'top_p': 0.95, 'n_predict': 400, 'stop': ["</s>", "Assistant:", "```", "User:"]}
    
class Local():
    def __init__(self, url, parameters=default_parameters):
        self.url = url
        
        self.context = "You are a friendly AI assistant designed to provide helpful, succinct, and accurate information."

        self.parameters = parameters
        
        return
    
    def get_server_health(self):
        response = requests.get(f'{self.url}/health')
        return response.json()
    
    def update_temp(self, temp):
        self.parameters['temperature'] = temp
    def update_k(self, k):
        self.parameters['top_k'] = k
    def update_p(self, p):
        self.parameters['top_p'] = p
    def update_n(self, n):
        self.parameters['n_predict'] = n
        
    def update_context(self, user_input, response):
        self.context = f'{self.context}\nUser: {user_input}\nAssistant:{response}'
        
    def clear_context(self):
        self.context = "You are a friendly AI assistant designed to provide helpful, succinct, and accurate information."
        
    def set_context(self, context):
        self.context = context
    
    def assistant_response(self, user_input):
        prompt = f"{self.context}\nUser: {user_input}\nAssistant:"
        self.parameters['prompt'] = prompt
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(f'{self.url}/completion', json=self.parameters, headers=headers)
        
        if response.status_code == 200:
            self.update_context(user_input, response.json()['content'].strip())
            return response.json()['content'].strip()
        else:
            return "Error processing your request. Please try again."
        
    def opencall(self):
        
        user_input = input("Enter a prompt or type 'exit' to quit: ")
        if self.get_server_health().get('status') != 'ok':
            print("Server not responding")
            return
    
        while user_input!="exit":
            print("Assistant Response:", self.assistant_response(user_input))
            user_input = input("Enter a prompt or type 'exit' to quit: ")
        
        return

    def prompt(self, prompt):
        self.parameters['prompt'] = prompt
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'{self.url}/completion', json=self.parameters, headers=headers)
        
        if response.status_code == 200:
            return response.json()['content'].strip()
        else:
            return "Error processing your request. Please try again."
        
from openai import OpenAI

# Default parameters for the ChatGPT API
default_chatGPT_parameters = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 450,
    "stop": ["</s>", "Assistant:", "User:"]
}

class ChatGPT():
    def __init__(self, api_key, parameters=default_chatGPT_parameters):
        self.client = OpenAI(api_key=api_key)
        self.context = "You are a friendly AI assistant designed to provide helpful, succinct, and accurate information."
        self.parameters = parameters
        self.model = 'gpt-4'

    def update_temp(self, temp):
        self.parameters['temperature'] = temp
        
    def update_p(self, p):
        self.parameters['top_p'] = p
        
    def update_n(self, n):
        self.parameters['max_tokens'] = n

    def update_context(self, user_input, response):
        self.context = f'{self.context}\nUser: {user_input}\nAssistant: {response}'

    def clear_context(self):
        self.context = "You are a friendly AI assistant designed to provide helpful, succinct, and accurate information."

    def set_context(self, context):
        self.context = context

    def assistant_response(self, user_input):
        try:
            # Build the prompt from context and user input
            messages = [{"role": "system", "content": self.context}, 
                        {"role": "user", "content": user_input}]
            
            # Make the API call
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=self.parameters["temperature"],
                top_p=self.parameters["top_p"],
                max_tokens=self.parameters["max_tokens"],
                stop=self.parameters["stop"]
            )
            
            # Extract the assistant's response
            assistant_reply = response["choices"][0]["message"]["content"].strip()
            
            # Update context
            self.update_context(user_input, assistant_reply)
            
            return assistant_reply
        
        except Exception as e:
            return f"Error processing your request: {e}"

    def open_call(self):
        user_input = input("Enter a prompt or type 'exit' to quit: ")
        while user_input.lower() != "exit":
            print("Assistant Response:", self.assistant_response(user_input))
            user_input = input("Enter a prompt or type 'exit' to quit: ")

    def prompt(self, prompt):
        """
        Feed a freeform prompt directly to ChatGPT without roles.
        
        :param prompt: The input text to be fed directly to the model.
        :return: The model's response as a string.
        """
        try:
            # Make the API call with the freeform prompt
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
                temperature=self.parameters["temperature"],
                top_p=self.parameters["top_p"],
                max_tokens=self.parameters["max_tokens"],
                stop=self.parameters["stop"]
            )
            
            # Return the response content
            return response.choices[0].message.content #.strip()
        
        except Exception as e:
            return f"Error processing your request: {e}"
        
        except Exception as e:
            return f"Error processing your request: {e}"
        
    def message(self, messages):
        response = self.client.chat.completions.create(
            model = self.model,
            messages=messages,
            temperature=self.parameters["temperature"],
            top_p=self.parameters["top_p"],
            max_tokens=self.parameters["max_tokens"]
            #stop=self.parameters["stop"]
            )
        return response.choices[0].message.content

class Agent():
    def __init__(self, model=Local(url='http://localhost:8080'), functions={}):
        
        self.agent = model
        
        self.tool_keys = list(functions.keys())
        
        self.functions = functions
        
        self.test_out = '' # For debugging purposes
        
        tools = ''
        
        for k in self.tool_keys:
            tools += (k + ': ' + functions[k].description + '\n')
        
        self.SYSTEM_PROMPT = f"""
You are a determined AI assistant designed to provide helpful, succinct, and accurate information.

Answer the following questions and obey the following commands as best you can.

You have access to the following tools:

{tools}Response To Human: When you need to respond to the human you are talking to.

You will receive a task from a person in the format "User: their task", then you should start a loop and do one of two things

Option 1: You use a tool to complete the task.
For this, you should use the following format:
Thought: Give yourself better insight into the task, this is not required
Action: The action to take, should be one of {self.tool_keys}
Input: "the input to the action, to be sent to the tool" (for multiple inputs separate by a comma e.g. "/path", "name", "print(This is a test)\\n" all inputs must be in quotes) Be sure that all escape codes are written out in code e.g \\n, \\t, do not use the acutal new line or tab escape characters

After this, the human will respond with an observation, and you will continue.

Option 2: You respond to the human. Do this once you are finished with the task
For this, you should use the following format:
Action: Response To Human
Input: "your response to the human"

Begin!
        """
        
        self.prompt = self.SYSTEM_PROMPT
        
        self.messages = [] # Again bc openAI must make everything complicated
        
        self.delta_temp = 0
        
        return
    
    def set_delta_temp(self, dt):
        self.delta_temp = dt
    
    def extract_action_and_input(self, text):
          action_pattern = r"Action: (.+?)\n"
          input_pattern =r"Input: (.+?)\n"
          action = re.findall(action_pattern, text)
          action_input = re.findall(input_pattern, text)
          return action, action_input
    
    def clean_output(self, input_string):
        lines = input_string.splitlines()
        self.test_out = lines
        print(lines)
        
        thought = ''
        action = ''
        input = ''
        
        for string in lines:
            if 'Thought:' in string:
                thought = string + '\n'
                break
        
        for string in lines:
            if 'Action:' in string:
                action = string + '\n'
                break
            
        for string in lines:
            if 'Input:' in string:
                input = string + '\n'
                break
            
        outstring = thought + action + input
        
        if outstring:
            action, action_input = self.extract_action_and_input(outstring)
            if not action or not action_input:
                return ''

        return outstring
    
    import re

    def parse_complex_input(self, input_string):
        # will separate each test "input" into their own list of strings that will get passed to a function
        return re.findall(r'"((?:[^"\\]|\\.)*?)"', input_string)
    
    def deleteMessage(self, num):
        # Construct full directory path
        folder_path = r"D:/Digit_recognizer_AI_stuff/LLM_Python_Folder/.removed"
        file_path = os.join(folder_path, f"Message_{num}")
        # Write to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.messages[num])
        
        self.messages[num] = {"role": self.messages[num]["role"], "content": "MESSAGE REMOVED"}
        
        return "Message Removed"
    
    def task_allocate(self, user_input, verbose=False):
        self.prompt = f"{self.prompt}\nUser: {user_input}\nAssistant:"
        if verbose:
            print(self.prompt)
        
        while True:
            tries = 0
            valid = False
            while not valid and tries < 5:
                response_text = self.clean_output(self.agent.prompt(self.prompt))
                valid = bool(response_text)
                if not valid:
                    self.agent.update_temp(self.delta_temp + self.agent.parameters['temperature'])
                tries += 1
                
            if not valid and tries >= 5: break
            
            if verbose:
                print(response_text)
            
            action, action_input = self.extract_action_and_input(response_text)
            
            task = self.parse_complex_input(action_input[0])
            
            if action[0] == "Response To Human":
                self.prompt = f"{self.prompt}\nAssistant: {response_text}"
                print(f"Response: {action_input[0]}\n")
                break
            elif action[0] in self.functions:
                observation = self.functions[action[0]].forward(*task)
            else:
                observation = "Action does not exist, please try again\n"
            
            if verbose:
                print("User: ", observation)
                
            self.prompt = f"{self.prompt}\nAssistant: {response_text}\nUser: {observation}\nAssistant:"
        
        return tries < 5
    
    
    
    def task_allocate_for_specifically_chatGPT_bc_openAI_sucks(self, user_input, verbose=False):
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT},
                         {"role": "user", "content": user_input}]
        
        if verbose:
            m_keys = list(self.messages[0].keys())  
            print(f'{self.messages[0][m_keys[0]]}: {self.messages[0][m_keys[1]]}')
            m_keys = list(self.messages[1].keys())  
            print(f'{self.messages[1][m_keys[1]]}: {self.messages[1][m_keys[1]]}')
        
        while True:
            tries = 0
            valid = False
            while not valid and tries < 5:
                response_text = self.clean_output(self.agent.message(self.messages))
                valid = bool(response_text)
                if not valid:
                    self.agent.update_temp(self.delta_temp + self.agent.parameters['temperature'])
                tries += 1
            
            if not valid and tries >= 5: break
        
            if verbose:
                print(response_text)
                
            action, action_input = self.extract_action_and_input(response_text)
            
            task = self.parse_complex_input(action_input[0])
            
            if action[0] == "Response To Human":
                self.prompt = f"{self.prompt}\nAssistant: {response_text}"
                print(f"Response: {action_input[0]}\n")
                break
            if action[0] == "Delete Message":
                observation = self.deleteMessage(*task)
            elif action[0] in self.functions:
                observation = self.functions[action[0]].forward(*task)
            else:
                observation = "Action does not exist, please try again\n"
            
            if verbose:
                print("User: ", observation)
                
            self.messages.append({"role": "system", "content": response_text})
            self.messages.append({"role": "user", "content": observation})
        
        return tries < 5
    
    def open_call(self):
        user_input = input("State a task or type 'exit' to quit: ")
        
        while user_input != 'exit':
            self.task_allocate(user_input)
            user_input = input("State a task of type 'exit' to quit: ")
        
        return


########## Functions #############
class google_search():
    def __init__(self, API_KEY, CSE_ID):
        os.environ["GOOGLE_CSE_ID"] = CSE_ID
        os.environ["GOOGLE_API_KEY"] = API_KEY
        
        self.cx = os.environ["GOOGLE_CSE_ID"]
        self.api = os.environ["GOOGLE_API_KEY"]
        
        self.description = "useful for when you need to answer questions about current events. You should ask targeted questions."
        return
    
    def forward(self, search_term):
        search_result = ""
        service = build("customsearch", "v1", developerKey=self.api)
        res = service.cse().list(q=search_term, cx=self.cx, num = 10).execute()
        for result in res['items']:
            search_result = search_result + result['snippet']
        return search_result

from EqSolver import Parser
class calculator():
    def __init__(self):
        self.description = "Useful for when you need to answer questions about math. Use python code, eg: 2 + 2"
        self.parser = Parser()
        return
    
    def forward(self, eq):
        return self.parser.parse(eq).evaluate({})

