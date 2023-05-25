import math
from typing import List


class Parser:
    def __init__(self, operands: List[str], constants: dict, operators: dict, functions: dict, mapping: dict):
        self.operands = operands
        self.constants = list(constants.keys())
        self.operators = list(operators.keys())
        self.functions = list(functions.keys())

        self.precedence = dict(operators, **functions)
        self.mapping = dict(constants, **mapping)

    def rpn_to_infix(self, rpn):
        rpn = rpn.split()
        stack = []

        for token in rpn:
            if token in self.operators and token in self.functions:
                if len(stack) == 2:
                    operand2 = str(stack.pop())
                    operand1 = str(stack.pop())
                    infix = '( ' + operand1 + ' ' + token + ' ' + operand2 + ' )'
                    stack.append(infix)

                elif len(stack) == 1:
                    operand = str(stack.pop())
                    infix = '( ' + token + ' ' + operand + ' )'
                    stack.append(infix)

                else:
                    stack.append(token)

            elif token in self.operators:
                operand2 = str(stack.pop())
                operand1 = str(stack.pop())
                infix = '( ' + operand1 + ' ' + token + ' ' + operand2 + ' )'
                stack.append(infix)

            elif token in self.functions:
                operand = str(stack.pop())
                infix = '( ' + token + ' ' + operand + ' )'
                stack.append(infix)

            else:
                stack.append(token)

        return stack[0]

    def infix_to_rpn(self, infix):  # sin ( x + y )
        infix = infix.split()
        stack = []
        output = []

        for token in infix:
            if token in self.operands or token in self.constants:
                output.append(token)

            elif token in self.precedence:
                while stack and stack[-1] != '(' and self.precedence[token] <= self.precedence[stack[-1]]:
                    output.append(stack.pop())
                stack.append(token)

            elif token == '(':
                stack.append(token)

            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop()

        while stack:
            output.append(stack.pop())

        return output


class SymbolicRegressor:
    def __int__(self, parser: Parser, variabless: List[dict[str:float]], targets: List[float]):
        self.parser = parser
        self.targets = targets
        self.variabless = variabless

    def evaluate(self, expression, variables: dict[str:float]):
        return eval(expression, {'__builtins__': None}, dict(variables, **self.parser.mapping))

    def rss(self, expression_values):
        return - sum([(expression_values[i] - self.targets[i]) ** 2 for i in range(len(self.targets))])




parser = Parser(operands=['x', 'y', 'z'],
                constants={'e': math.e, 'pi': math.pi},
                operators={'+': 1, '-': 1, '*': 2, '/': 2},
                functions={'sin': 3, 'cos': 3, '-': 1},
                mapping={'sin': math.sin, 'cos': math.cos})

r = '- sin ( cos ( - x + y - z + e ) )'
print(r)

r = parser.infix_to_rpn(r)
print(r)

r = ' '.join(r)
print(r)

r = parser.rpn_to_infix(r)
print(r)

r = parser.infix_to_rpn(r)
print(r)

r = ' '.join(r)
print(r)

r = parser.rpn_to_infix(r)
print(r)

m = parser.evaluate(r)
print(m)
