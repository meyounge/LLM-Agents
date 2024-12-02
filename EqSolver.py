# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 04:10:08 2024

@author: chatGPT
"""

import ast
import operator

class Parser:
    def __init__(self):
        # Supported operations
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }

    def parse(self, expression):
        # Parse expression into AST
        self.expr = expression
        self.tree = ast.parse(expression, mode='eval').body
        return self

    def evaluate(self, _):
        # Evaluate AST recursively
        return self._eval(self.tree)

    def _eval(self, node):
        if isinstance(node, ast.BinOp):
            # Binary operation node (e.g., 2 + 3)
            left = self._eval(node.left)
            right = self._eval(node.right)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.Constant):  # Updated to ast.Constant
            # Constant node for numbers
            return node.value  # Updated to node.value
        elif isinstance(node, ast.Expression):
            # Expression node (top-level in AST for eval mode)
            return self._eval(node.body)
        else:
            raise ValueError(f"Unsupported operation: {ast.dump(node)}")

#Calculator
parser = Parser()
def calculator(eq):
    return parser.parse(eq).evaluate({})

print(calculator("5*5+4"))