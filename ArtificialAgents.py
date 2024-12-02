# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:20:20 2024

@author: Michael Younger
"""
import requests
import os, re
from openai import OpenAI
import google.generativeai as genai
from abc import abstractmethod
import tiktoken
import csv
from io import StringIO


universal_parameters = {'temp': 0.8, 'top_k': 35, 'top_p': 0.95, 'tokens': 1000, 'stop': ["</s>"]}
universal_context = "You are a friendly AI assistant designed to provide helpful, succinct, and accurate information."
API_KEYS = {'Gemini': '','chatGPT': '', 'Local': ''}

class LLM():
    @abstractmethod
    def __init__(self, parameters=universal_parameters, context=universal_context):
        self.context = context
        self.parameters = parameters
        self.key = {'role': 'role', 'content': 'content'}
        self.roles = {'AI': 'assistant', 'user': 'user', 'system': 'system'}
        return
    def setparameters(self, parameters):
        self.paramters = parameters
        return
    def setcontext(self, context):
        self.context = context
        return
    def settokens(self, tokens):
        self.parameters['tokens'] = tokens
        return
    def setstop(self, stoplist):
        self.parameters['stop'] = stoplist
        return
    def appendstop(self, stoplist):
        self.parameters['stop'] += stoplist
        return
    def prompt(self, prompt):
        return "This function will return a zero shot response to an LLM"
    def message(self, messages):
        output = """
        This function will take in a list of dictionaries that are in the form of "role": "their role", "content": "The content"
        It will then process this and get a reponse from the AI and returns it
        """
        return output
    def opencall(self):
        user_input = input("Enter a prompt or type 'exit' to quit: ")
        messages = [{self.key['role']: self.roles['AI'], self.key['content']: self.context},
                    {self.key['role']: self.roles['user'], self.key['content']: user_input}]
        
        while user_input != 'exit':
            response = self.message(messages)
            print(response)
            user_input = input("Enter a prompt or type 'exit' to quit: ")

            messages.append({self.key['role']: self.roles['AI'], self.key['content']: response})
            messages.append({self.key['role']: self.roles['user'], self.key['content']: user_input})
        return

class Local(LLM):
    def __init__(self, url, parameters=universal_parameters, context=universal_context):
        super().__init__(parameters, context)
        self.url = url
        return
    def getserverhealth(self):
        return requests.get(f'{self.url}/health').json()
    def prompt(self, prompt):
        sendjson = self.parameters.copy()
        sendjson['prompt'] = prompt
        sendjson['n_predict'] = sendjson.pop('tokens')
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'{self.url}/completion', json=sendjson, headers=headers)
        
        if response.status_code == 200:
            return response.json()['content'].strip()
        else:
            return f"Error processing your request. status code: {response.status_code}."
    def message(self, messages):
        prompt = ""
        for m in messages:
            prompt += (m[self.key['role']] + ': ' + m[self.key['content']] + '\n')
        return self.prompt(prompt)

class chatGPT(LLM):
    def __init__(self, model, api_key=API_KEYS['chatGPT'], parameters=universal_parameters, context=universal_context):
        super().__init__(parameters, context)
        self.client = OpenAI(api_key=api_key)
        self.model = model
        return
    
    def tokenAmount(self, text):
        encoding = tiktoken.encoding_for_model(self.model)
        return encoding.encode(text)
    
    def prompt(self, prompt):
        try: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=self.parameters["temp"],
                top_p=self.parameters["top_p"],
                max_tokens=self.parameters["tokens"],
                stop=self.parameters["stop"]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing your request: {e}"
    def message(self, messages):
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages=messages,
                temperature=self.parameters["temp"],
                top_p=self.parameters["top_p"],
                max_tokens=self.parameters["tokens"],
                stop=self.parameters["stop"]
                )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing your request: {e}"

class Gemini(LLM):
    def __init__(self, model, api_key, parameters=universal_parameters, context=universal_context):
        super().__init__(parameters, context)
        
        os.environ["Gemini_API_KEY"] = api_key
        genai.configure(api_key=os.environ["Gemini_API_KEY"])
        
        self.model = genai.GenerativeModel(model)
        
        self.key =  {'role': 'role', 'content': 'parts'}
        self.roles = {'AI': 'model', 'user': 'user', 'system': 'model'}
        
        self.chat = []
        
        return
    
    def prompt(self, prompt):
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.parameters['tokens'],
                    temperature=self.parameters['temp'],
                    top_p=self.parameters['top_p'],
                    top_k=self.parameters['top_k'],
                    stop_sequences = self.parameters['stop']
                ),
            )
            return response.text
        except Exception as e:
            return f"Error processing your request: {e}"
        
    
    def message(self, messages):
        self.chat = self.model.start_chat(history=messages[:-1])
        try:
            response = self.chat.send_message(messages[-1]['parts'],
                        generation_config=genai.GenerationConfig(
                            max_output_tokens=self.parameters['tokens'],
                            temperature=self.parameters['temp'],
                            top_p=self.parameters['top_p'],
                            top_k=self.parameters['top_k'],
                            stop_sequences = self.parameters['stop']
                        ),
                    )
            return response.text
        except Exception as e:
            return f"Error processing your request: {e}"

class Agent():
    def __init__(self, model, functions):
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

{tools}Final Response: Final output after the task is completed


You will receive a task from a user in the format "user: their task", then you should start a loop and do one of two things

Option 1: You use a tool to complete the task.
For this, you should use the following format:
Thought: Give yourself better insight into the task, this is not required
Action: The action to take, should be one of {self.tool_keys}
Input: The input to the function: [input1, input2, input3, etc.] *ALL INPUTS MUST BE BRACKETED

IMPORTANT NOTE: All code inputs (like writing to a python file!) should be surrounded with 3 backticks on both sides like so: 
    ```print('hello world')```, failure to do so will result in undefined behavior on the parser

After this, the user will respond with an observation, and you will continue.

Option 2: You are finished with the task and would like to end the sequence.
For this, you should use the following format:
Action: Final Response
Input: [your response to the user as confirmation of task completion]

IMPORTANT NOTE 2: The parser for this system is still in early development, so it can only handle one action and input at a time, so please only give one response at a time!

Begin!
    	"""
        
        self.messages = [{self.agent.key['role']: self.agent.roles['system'], self.agent.key['content']: self.SYSTEM_PROMPT}]
        
        return
    
    def changesystemprompt(self, new_prompt):
        self.messages[0] = {self.agent.key['role']: self.agent.roles['system'], self.agent.key['content']: new_prompt}
        return
    
    def parser(self, text):
        #text = text.encode('utf-8').decode('unicode_escape')
        
        # Define regex patterns for Thought, Action, and Input
        thought_pattern = r"Thought:\s*(.*?)\n"
        action_pattern = r"Action:\s*(.*?)\n"
        input_pattern_list = r"Input:\s*\[(.*?)\]"
        input_pattern_code_1 = r"\s*```python\\n(.*?)```"
        input_pattern_code_2 = r"\s*```(.*?)```"
        
        # Extract matches
        thought_match = re.search(thought_pattern, text)
        action_match = re.search(action_pattern, text)
        
        code_list = re.findall(input_pattern_code_1, text, re.DOTALL)
        Extra = ''
        if code_list:
            Extra = 'python\\n'
        else:
            code_list = re.findall(input_pattern_code_2, text, re.DOTALL)
        
        print(code_list)
        
        # Extract and clean Thought and Action
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
    
        # Temporarily replace code blocks with placeholders
        input_text = text
        for code in code_list:
            input_text = input_text.replace(f"```{Extra}{code}```", "*code")
            
        print(input_text)
    
        # Extract the Input section
        input_match = re.search(input_pattern_list, input_text, re.DOTALL)
        if not input_match:
            return thought, action, None
        
        # Use csv to correctly parse the list
        input_str = input_match.group(1).strip()
        input_io = StringIO(input_str.replace("\n", ""))  # Ensure newlines are preserved correctly
        reader = csv.reader(input_io, skipinitialspace=True)
        parsed_input = next(reader)  # Parse the single line of CSV into a list
        
        #STRIP EVERYTHING
        for i, item in enumerate(parsed_input):
            parsed_input[i] = item.strip("'\"")
        
        # Replace placeholders with actual code blocks
        for i, item in enumerate(parsed_input):
            if item == "*code":
                parsed_input[i] = code_list.pop(0).strip("'\"")
                
        return thought, action, parsed_input
    
    def processinput(self, action, task):
        if action in self.functions:
            try:
                output = self.functions[action].forward(*task)
            except Exception as e:
                print(f"EXCEPTION {e} CAUGHT")
                print("Action: ", action)
                print("Task List: ", task)
                output = input("Please describe the issue: ")
                
            return output
        else:
            return "Action does not exist, please try again\n"
    
    def taskallocate(self, user_input, verbose=False):
        
        if verbose:
            print(self.messages[-1][self.agent.key['content']])
            print("user:\n", user_input)
            
        self.messages.append({self.agent.key['role']: self.agent.roles['user'], self.agent.key['content']: user_input})
        
        while True:
            tries = 0
            while tries < 5:
                response_text = self.agent.message(self.messages)
                self.test_out = response_text
                if 'Action:' in response_text and 'Input:' in response_text:
                    break
                if tries == 2:
                    self.messages.append({self.agent.key['role']: self.agent.roles['system'], self.agent.key['content']: "The output must in the format: \nThought: \nAction: \nInput: \nFailure to do so will result in an invalid response, this is actually your third try responding so get it right this time"})
                tries += 1
            
            if tries >= 5: break
            
            self.messages.append({self.agent.key['role']: self.agent.roles['AI'], self.agent.key['content']: response_text}) # This should happen immedietly after a valid response has returned
            
            if verbose:
                print(response_text)
                
            thought, action, input_ = self.parser(response_text)
            
            
            if action == "Final Response":
                self.messages.append({self.agent.key['role']: self.agent.roles['AI'], self.agent.key['content']: response_text})
                return input_[0]
            
            observation = self.processinput(action=action, task=input_)
            
            if verbose:
                print("User: ", observation)
                
            self.messages.append({self.agent.key['role']: self.agent.roles['user'], self.agent.key['content']: observation})
        
        return "I am sorry, I failed to do the task required of me, please let me try again by re-prompting me with modified instructions"

class LeadAgent(Agent):
    def __init__(self, model, functions):
        super().__init__(model, functions)
        
        self.AgentDict = {}
        
        tools = ''
        for k in self.tool_keys:
            tools += (k + ': ' + functions[k].description + '\n')
        
        llmlist = []
        for k in list(API_KEYS.keys()):
            if API_KEYS[k] != '':
                llmlist.append(k)
        
        llmdescriptions = []
        if 'Local' in llmlist:
            llmdescriptions.append('Local is a local AI that is good for small tasks, it can be slow, but it is inexpensive to run')
        if 'chatGPT' in llmlist:
            llmdescriptions.append('chatGPT is a Large Language Model developed by OpenAI, it is fast and good for a wide variery of tasks')
        if 'Gemini' in llmlist:
            llmdescriptions.append('Gemini is an Large Language Model developed by Google with an astoundingly large context window and fast results')
        
        self.SYSTEM_PROMPT = f"""
You are a lead AI assistant designed to designate tasks to other AI agents in order to complete complex tasks

Answer the following questions and obey the following commands as best you can.

You have access to the unique function:
Agent Task:
    This function will allow you to utalize another AI agent in order to complete a task
    
    Args:
        agent_type (str): The LLM agents you can utilize are {llmlist} {llmdescriptions}
        agent_name (str): A name for the agent you define; if you want to call upon the same agent again, ex . 'Test Manager' or 'Script Writer' ; 
                            Understand that conversations you have with any agent are not global, so be sure to re-mention important details between agents
        task_assignment (str): The task that the AI is assigned to do, be sure to be as clear as you can
        
    Returns:
        str: The agent's response after completing the task; if the agent failed to do so it will tell you as well, please retry with another prompt

Your AI subordinates have access to the following tools:

{tools}Final Response: Final output after the task is completed

You also have access to the above tools, but be sure to use them sparingly as you should rely on your subordinates for most tasks

You will receive a complex task from a user in the format "user: their task", then you should start a loop and do one of three things

Option 1: You use a tool to complete the task.
For this, you should use the following format:
Thought: Give yourself better insight into the task, this is not required
Action: The action to take, should be one of {self.tool_keys}
Input: The input to the function: [input1, input2, input3, etc.]

IMPORTANT NOTE: All code inputs (like writing to a python file!) should be surrounded with 3 backticks on both sides like so: 
    ```print('hello world')```, faliure to do so will result in undefined behavior on the parser

After this, the user will respond with an observation, and you will continue.

Option 2: You dedicate an agent to a task. (Be sure to not overburden a single agent)
For this, you should use the following format:
Thought: Give yourself better insight into the task, this is not required
Action: Agent Task
Input: ["one of {llmlist}", "Their name", "Their task"]

After this, you should continue to the next step of the complex task

Option 3: You are finished with the task and would like end the sequence.
For this, you should use the following format:
Action: Final Response
Input: [your response to the user as confirmation of task completion]

IMPORTANT NOTE 2: The parser for this system is still in early development, so it can only handle one action and input at a time, so please only give one reponse at a time!

Begin!
        """
        
        self.messages = [{self.agent.key['role']: self.agent.roles['system'], self.agent.key['content']: self.SYSTEM_PROMPT}]
        
    def processinput(self, action, task):
        if action == 'Agent Task':
            if len(task) != 3:
                print("ISSUE ON INPUTS")
                print("Action: ", action)
                print("Task List: ", task)
                output = input("Please describe the issue: ")
                return output
            a_type, a_name, content = task
            
            if a_name in self.AgentDict:
                return self.AgentDict[a_name].taskallocate(content)
            else:
                if a_type == 'Local':
                    self.AgentDict[a_name] = Agent(model = Local(url=API_KEYS['Local']), functions=self.functions)
                elif a_type == 'Gemini':
                    self.AgentDict[a_name] = Agent(model=Gemini(model="gemini-1.5-flash", api_key=API_KEYS['Gemini']), functions=self.functions)
                else:
                    self.AgentDict[a_name] =  Agent(model = chatGPT(api_key=API_KEYS['chatGPT'], model='gpt-4'), functions=self.functions)
            
                return self.AgentDict[a_name].taskallocate(content)
            
        elif action in self.functions:
            try:
                output = self.functions[action].forward(*task)
            except Exception as e:
                print(f"EXCEPTION {e} CAUGHT")
                print("Action: ", action)
                print("Task List: ", task)
                output = input("Please describe the issue: ")
                
            return output
        else:
            return "Action does not exist, please try again\n"
        
#################### AGENT FUNCTIONS ####################

class AgentFunction():
    @abstractmethod
    def __init__(self):
        self.description = "This function does not have an initialized description (blame the programmer), please assume what it does from its name"
        return
    def forward():
        return "This function is what will be done when the AI calls upon it"
    

from googleapiclient.discovery import build
class google_search(AgentFunction):
    def __init__(self, API_KEY, CSE_ID):
        os.environ["GOOGLE_CSE_ID"] = CSE_ID
        os.environ["GOOGLE_API_KEY"] = API_KEY
        
        self.cx = os.environ["GOOGLE_CSE_ID"]
        self.api = os.environ["GOOGLE_API_KEY"]
        
        self.description = """
    Google search for things on the internet and get a snippet that search back.

    Args:
        search_term (str): The thing you want to look up 

    Returns:
        str: The snippet of the google search information
        """
        return
    
    def forward(self, search_term):
        search_result = ""
        service = build("customsearch", "v1", developerKey=self.api)
        res = service.cse().list(q=search_term, cx=self.cx, num = 10).execute()
        for result in res['items']:
            search_result = search_result + result['snippet']
        return search_result

from EqSolver import Parser
class calculator(AgentFunction):
    def __init__(self):
        self.description = """
    Is a python calculator.

    Args:
        equation (str): A python syntax equation to evaluate

    Returns:
        str: The calculated output
        """
        self.parser = Parser()
        return
    
    def forward(self, equation):
        return self.parser.parse(equation).evaluate({})

class WriteFile(AgentFunction):
    def __init__(self, home_dir):
        
        self.home_dir = home_dir
        self.description = """
    Writes the provided content to a file, creating necessary directories if they don't exist.

    Args:
        file_name (str): The name of the file (the file extension will indicate the type of file created e.g. .py, .txt).
        content (str): The content to write into the file.
        directory (str, optional): The relative directory within the home directory where the file should be saved. Defaults to './'.

    Returns:
        str: A status message indicating the file was successfully saved.
        """

    def reWrite_EscapeSequence(self, input_string):
        return input_string.encode('utf-8').decode('unicode_escape')
        
    def forward(self, file_name, content, directory=''):

        if directory == '/':
            return "WARN: Use './' not '/' when saving to home directory, file not saved, please try again"
        
        # Construct full directory path
        full_path = os.path.join(self.home_dir, directory)
        os.makedirs(full_path, exist_ok=True)  # Ensure the directory exists
        
        # Full file path
        file_path = os.path.join(full_path, file_name)
        
        # Write to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.reWrite_EscapeSequence(content))
        
        return "The file was successfully saved."

class AppendFile(AgentFunction):
    def __init__(self, home_dir):
        self.home_dir = home_dir
        self.description = """
    Appends the provided content to an existing file, creating the file and necessary directories if they don't exist.

    Args:
        file_name (str): The name of the file (the file extension will indicate the type of file created e.g. .py, .txt).
        content (str): The content to append to the file.
        directory (str, optional): The relative directory within the home directory where the file should be saved. Defaults to './'.


    Returns:
        str: A status message indicating the file was successfully saved.
        """
    def reWrite_EscapeSequence(self, input_string):
        return input_string.encode('utf-8').decode('unicode_escape')
    def forward(self, file_name, content, directory=''):
        
        if directory == '/':
            return "WARN: Use './' not '/' when saving to home directory, file not saved, please try again"
        
        # Construct full directory path
        full_path = os.path.join(self.home_dir, directory)
        os.makedirs(full_path, exist_ok=True)  # Ensure the directory exists
        
        # Full file path
        file_path = os.path.join(full_path, file_name)
        
        # Append to the file
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(self.reWrite_EscapeSequence(content))
        
        return "The file was successfully saved."

class ListExistingFiles(AgentFunction):
    def __init__(self, home_dir):
        self.home_dir = home_dir
        
        self.description = """
    Lists the files and folders in the specified directory ('./' is treated as the home directory).
    Excludes dot-prefixed files from view, but they can still be accessed.
    Only lists top-level contents of the directory.

    Args:
        directory (str): The directory to list.
        search_text (str): Text to filter files. Use '*all' to list all files.
    
    Returns:
        list: A list of file names (not full paths) that match the criteria.
        """
    def listtostring(self, listofstrings):
        outstring = 'The files in the directory are:\n'
        for l in listofstrings:
            outstring += (l + '\n')
        return outstring
    
    def forward(self, directory, search_text):
        if './' not in directory:
            return "Invalid directory name (should start with ./)"
        
        if './.' in directory:
            hidden='.'
        else:
            hidden=''
        
        
        full_directory = os.path.normpath(
            os.path.join(self.home_dir, hidden + directory.lstrip('./'))
        )
        
        # Check if the directory exists
        if not os.path.exists(full_directory):
            raise FileNotFoundError(f"Directory '{full_directory}' does not exist.")
        
        # List only top-level contents of the directory
        try:
            items = os.listdir(full_directory)  # Surface-level listing only
        except Exception as e:
            return f"Could not access directory '{full_directory}': {e}"
        
        # Determine if the directory is hidden
        is_hidden_directory = os.path.basename(full_directory).startswith('.')
        
        # Filter items based on conditions
        filtered_items = []
        for item in items:
            # Skip dot-prefixed items unless we're inside a hidden directory
            if not is_hidden_directory and item.startswith('.'):
                continue
            
            item_path = os.path.join(full_directory, item)
            
            # Check if the item is a file and matches the search criteria
            if os.path.isfile(item_path):
                if search_text == "*all" or (search_text in item):
                    filtered_items.append(item)
            elif os.path.isdir(item_path) and search_text == "*all":
                # Include directories only when listing all items
                filtered_items.append(item)
        
        return self.listtostring(filtered_items)
    
class CreateDirectory(AgentFunction):
    def __init__(self, home_dir):
        self.home_dir = home_dir
        self.description = """
    Creates a new directory at the specified path. If the directory already exists, it does nothing.
    
    Args:
        path (str): The path where the directory should be created.
        
    Returns:
        str: Confirmation that the directory was created and where
    """
        
    def forward(self, directory):
        path = os.path.join(self.home_dir, directory)
        os.makedirs(path, exist_ok=True)
        
        return f"Created directory at /{directory}"
    
class ReadFileContents(AgentFunction):
    def __init__(self, home_dir):
        self.description = """
    Reads the contents of a specified file.

    Args:
        file_name (str): The name of the file
        directory (str, optional): The relative directory within the home directory where the file is located. Defaults to './'.

    Returns:
        str: The contents of the file as a string. If the file does not exist, an appropriate error message is returned.
        """
        self.home_dir = home_dir

    def forward(self, file_name, directory=''):
        file_name = file_name.strip("'\"")
        
        full_path = os.path.join(self.home_dir, directory, file_name)
        
        try:
            with open(full_path, "r", encoding="utf-8") as file:
                content = file.read()
                if content == '':
                    return "The file is empty"
                else:
                    return content
        except FileNotFoundError:
            return f"The file '{file_name}' does not exist in the directory '{directory}'."
        except Exception as e:
            return f"An error occurred while reading the file: {str(e)}"

import subprocess
class ExecuteScript(AgentFunction):
    def __init__(self, home_dir):
        self.description = """
    Executes a Python script and redirects its output to a file.

    Args:
        script_name (str): Name of the Python script to execute.
        output_name (str): Name of the output file to save the script's output.
        directory (str, optional): Directory containing the script. Defaults to './'.
        output_directory (str, optional): Directory to save the output file. Defaults to './'.

    Returns:
        str: Full path of the output file or an error message.
        """
        self.home_dir = home_dir

    def forward(self, script_name, output_name, directory='./', output_directory='./'):
        
        if directory == '/' or output_directory == '/':
            return "Error: Use './' not '/' when referencing the home directory"
        
        # Determine the script path
        script_path = os.path.join(self.home_dir, directory, script_name)
        if not os.path.isfile(script_path):
            return f"Error: Script '{script_name}' not found in '{directory}'."

        # Determine the output file path
        output_path = os.path.join(self.home_dir, output_directory, output_name)

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Execute the script and redirect output
            with open(output_path, 'w') as output_file:
                subprocess.run(['python', script_path], stdout=output_file, stderr=subprocess.STDOUT, check=True)

            return f"Script executed successfully. Output saved to '{os.path.join(output_directory, output_name)}'."
        except subprocess.CalledProcessError as e:
            print(f"Error {e} Triggered\nError during script execution: See {os.path.join(output_directory, output_name)} for more information.")
            return f"Error during script execution: See {os.path.join(output_directory, output_name)} for more information."
        except Exception as e:
            return f"Unexpected error: {e}."

class AskUser(AgentFunction):
    def __init__(self):
        self.description = """
    Ask a question to the user for clarification or further instruction

    Args:
        question (str): Your question to the user
        
    Returns:
        str: The user's response
        """
    def forward(self, question):
        print("AI Question: ", question)
        return input("User: ")

FILEPATH = '/'
default_function_list = {
    "Create Directory":CreateDirectory(FILEPATH), 
    "List Files in Directory": ListExistingFiles(FILEPATH), 
    "Write to File": WriteFile(FILEPATH),
    "Append File": AppendFile(FILEPATH),
    "Read File Contents": ReadFileContents(FILEPATH),
    "Execute Script": ExecuteScript(FILEPATH),
    "Ask Question": AskUser()
     }

