import math
import random
from typing import List


class Parser:
    def __init__(self, operands: List[str], constants: dict, operators: dict, functions: dict, mapping: dict):
        self.operands = operands
        self.constants = list(constants.keys())
        self.operators = list(operators.keys())
        functions_keys = list(functions.keys())
        self.functions = []
        self.replaces = []

        for function_key in functions_keys:
            if function_key in self.operators:
                self.functions.append('~' + function_key)
                self.replaces.append(function_key)
            else:
                self.functions.append(function_key)

        self.precedence = dict(operators, **functions)
        self.mapping = dict(constants, **mapping)

    def rpn_to_infix(self, rpn):
        rpn = rpn.split()
        stack = []

        for token in rpn:
            # if token in self.operators and token in self.functions:
            #     if len(stack) == 2:
            #         operand2 = str(stack.pop())
            #         operand1 = str(stack.pop())
            #         infix = f'( {operand1} {token} {operand2} )'
            #         stack.append(infix)
            #
            #     elif len(stack) == 1:
            #         operand = str(stack.pop())
            #         infix = f'( {token} {operand} )'
            #         stack.append(infix)
            #
            #     else:
            #         stack.append(token)

            if token in self.operators:
                operand2 = str(stack.pop())
                operand1 = str(stack.pop())
                infix = f'( {operand1} {token} {operand2} )'
                stack.append(infix)

            elif token in self.functions:
                operand = str(stack.pop())
                infix = f'( {token} {operand} )'
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
    def __init__(self, parser: Parser, variables, targets, population_size, max_depth, selection_size, max_generations, error):
        self.parser = parser
        self.targets = targets
        self.variables = variables
        self.selection_size = selection_size
        self.max_generations = max_generations
        self.error = error
        self.population_size = population_size

        self.population = self.generate_population(population_size, max_depth)

    def evaluate(self, expression, variables):
        try:
            return eval(expression, {'__builtins__': None}, dict(variables, **self.parser.mapping))
        except Exception:  # ZeroDivisionError
            return float('inf')

    def rss(self, expression):
        try:
            return - sum([abs(self.evaluate(expression, self.variables[i]) - self.targets[i]) for i in range(len(self.targets))])
        except:
            return float('-inf')

    def generate_expression(self, depth):
        if depth == 0:
            return random.choice(self.parser.operands + self.parser.constants)

        task = random.choice(self.parser.operators + self.parser.functions)

        if task in self.parser.operators:
            left = self.generate_expression(depth - 1)
            right = self.generate_expression(depth - 1)
            return f'( {left} {task} {right} )'

        else:
            value = self.generate_expression(depth - 1)
            return f'( {task} ( {value} ) )'

    def generate_population(self, population_size, max_depth):
        return [self.generate_expression(random.randint(1, max_depth)) for _ in range(population_size)]

    def get_selection(self):
        scored_expressions = [(self.rss(expression), expression) for expression in self.population]
        scored_expressions = sorted(scored_expressions, reverse=True, key=lambda x: x[0])
        return scored_expressions[:self.selection_size]

    def crossover(self, rpn_parent1, rpn_parent2):
        i = 0

        while i != 3:
            try:
                # parent1 = self.parser.infix_to_rpn(parent1)
                # parent2 = self.parser.infix_to_rpn(parent2)

                parent1_num = random.randint(1, len(rpn_parent1) - 2)
                parent2_num = random.randint(1, len(rpn_parent2) - 2)

                new_parent1 = rpn_parent1[:parent1_num] + rpn_parent2[parent2_num:]
                new_parent2 = rpn_parent2[:parent2_num] + rpn_parent1[parent1_num:]

                self.parser.rpn_to_infix(' '.join(new_parent1))
                self.parser.rpn_to_infix(' '.join(new_parent2))

                return new_parent1, new_parent2

            except:
                i += 1

        return None

    def evolve(self):
        best_score = None

        for generation in range(self.max_generations):
            selection = self.get_selection()

            best_score = selection[0]

            if -best_score[0] <= self.error and best_score[0] != float('inf'):
                return best_score

            print(f'Generation {generation}: min error = {best_score[0]}')

            new_population = [i[1] for i in selection]

            for i in range((self.population_size - self.selection_size) // 2):
                parent1 = self.parser.infix_to_rpn(random.choice(selection)[1])
                parent2 = self.parser.infix_to_rpn(random.choice(selection)[1])

                crossed = self.crossover(parent1, parent2)

                if not crossed:
                    parent1 = self.parser.rpn_to_infix(' '.join(self.mutate(parent1)))
                    parent2 = self.parser.rpn_to_infix(' '.join(self.mutate(parent2)))
                else:
                    parent1 = self.parser.rpn_to_infix(' '.join(self.mutate(crossed[0])))
                    parent2 = self.parser.rpn_to_infix(' '.join(self.mutate(crossed[1])))

                new_population.append(parent1)
                new_population.append(parent2)

            self.population = new_population

        return best_score

    def mutate(self, rpn_expression):
        rand = random.randint(0, len(rpn_expression) - 1)

        if rpn_expression[rand] in self.parser.operands or rpn_expression[rand] in self.parser.constants:
            return [(random.choice(self.parser.operands + self.parser.constants) if i == rand else rpn_expression[i]) for i in range(len(rpn_expression))]

        elif rpn_expression[rand] in self.parser.operators:
            return [(random.choice(self.parser.operators) if i == rand else rpn_expression[i]) for i in range(len(rpn_expression))]

        else:
            return [(random.choice(self.parser.functions) if i == rand else rpn_expression[i]) for i in range(len(rpn_expression))]


parser = Parser(operands=['x', 'y', 'z'],
                constants={'e': math.e},  # {'e': math.e, 'pi': math.pi}
                operators={'+': 1, '-': 1, '*': 2, '/': 2, '**': 3},
                functions={'sin': 4, 'cos': 4, '-': 4, 'log10': 4, 'ln': 4},
                mapping={'sin': math.sin, 'cos': math.cos, 'log10': math.log10, 'ln': lambda x: math.log(x, math.e)})

sr = SymbolicRegressor(parser=parser,
                       variables=[{'x': 3, 'y': 2, 'z': 5}, {'x': 1, 'y': 2, 'z': 3}, {'x': 0, 'y': 0, 'z': 1}],
                       targets=[2**2 + math.exp(10), 2**2 + math.exp(6), math.exp(1)],
                       population_size=10000,
                       max_depth=4,
                       selection_size=10000 // 2,
                       max_generations=100,
                       error=0.0001)

# e^(x + y + z)
best = sr.evolve()
print(best)
