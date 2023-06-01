import math
import random

import numpy as np
import gc


def fitness_function(expression, x_mappings, y):
    output = sum([np.abs(evaluate_tree(expression, x_mappings[i]) - y[i]) for i in range(len(y))]) / len(y)

    if math.isnan(output):
        return float('inf')

    return output


def evaluate_tree(node, x_mappings):
    if node.type == 'operand':
        return x_mappings[node.value]

    elif node.type == 'constant':
        return node.value

    elif node.type == 'unary_operator':
        left = evaluate_tree(node.left, x_mappings)
        return node.value(left)

    elif node.type == 'binary_operator':
        left = evaluate_tree(node.left, x_mappings)
        right = evaluate_tree(node.right, x_mappings)
        return node.value(left, right)

    raise (Exception(f'Invalid node type: {node.type}'))


def initialize_population(max_depth, population_size, unary_operators, binary_operators, operands, constants):
    population = []

    while len(population) < population_size:
        tree = generate_random_tree(max_depth, unary_operators, binary_operators, operands, constants)

        if tree not in population:
            population.append(tree)

    return population


def generate_random_tree(max_depth, unary_operators, binary_operators, operands, constants):
    if max_depth == 0:
        value = random.choice(operands + constants)
        if value in operands:
            return Node(value, type='operand')
        else:
            return Node(value, type='constant')
    else:
        if random.random() < 0.5:
            value = random.choice(operands + constants)
            if value in operands:
                return Node(value, type='operand')
            else:
                return Node(value, type='constant')
        else:
            value = random.choice(unary_operators + binary_operators)
            if value in unary_operators:
                return Node(value, generate_random_tree(max_depth - 1, unary_operators, binary_operators, operands, constants), None, 'unary_operator')
            else:
                return Node(value, generate_random_tree(max_depth - 1, unary_operators, binary_operators, operands, constants),
                            generate_random_tree(max_depth - 1, unary_operators, binary_operators, operands, constants), 'binary_operator')


def minus(x):
    return -x


class Node:
    mapping = None

    def __init__(self, value, left=None, right=None, type='operand'):
        self.value = value
        self.left = left
        self.right = right
        self.type = type

    def copy(self):
        if self.type == 'operand':
            return Node(self.value, type='operand')
        elif self.type == 'constant':
            return Node(self.value, type='constant')
        elif self.type == 'unary_operator':
            return Node(self.value, self.left.copy(), None, 'unary_operator')
        elif self.type == 'binary_operator':
            return Node(self.value, self.left.copy(), self.right.copy(), 'binary_operator')

    def __str__(self):
        if self.type == 'operand':
            return str(self.value)
        elif self.type == 'constant':
            return str(self.value)
        elif self.type == 'unary_operator':
            if self.value == minus:
                return f'(-({self.left}))'
            else:
                return f'{Node.mapping[self.value]}({self.left})'
        elif self.type == 'binary_operator':
            return f'({self.left} {Node.mapping[self.value]} {self.right})'

    def __eq__(self, other):
        if self.type == 'operand':
            return self.value == other.value
        elif self.type == 'constant':
            return self.value == other.value
        elif self.type == 'unary_operator':
            return self.value == other.value and self.left == other.left
        elif self.type == 'binary_operator':
            return self.value == other.value and self.left == other.left and self.right == other.right


def selection(population, fitness, selection_size):
    best_indices = np.argsort(fitness)[:selection_size]
    return [population[i] for i in best_indices], [fitness[i] for i in best_indices]


def crossover(parent1, parent2):
    paret1_copy = parent1.copy()
    paret2_copy = parent2.copy()

    if parent1.type in ['operand', 'constant'] or parent2.type in ['operand', 'constant']:
        return paret2_copy, paret1_copy

    nodes1 = get_tree_nodes(paret1_copy, need_root=False)
    node1, depth1 = random.choice(nodes1)

    nodes2 = get_tree_nodes_with_depth(paret2_copy, target=depth1)
    node2, depth2 = random.choice(nodes2)

    temp = node1.left
    node1.left = node2.left
    node2.left = temp

    temp = node1.right
    node1.right = node2.right
    node2.right = temp

    temp = node1.value
    node1.value = node2.value
    node2.value = temp

    temp = node1.type
    node1.type = node2.type
    node2.type = temp

    return paret1_copy, paret2_copy


def mutate(expression, max_depth, mutation_probability, unary_operators, binary_operators, operands, constants):
    node = random.choice(get_tree_nodes(expression))[0]

    if random.random() < mutation_probability:
        if node.type == 'operand':
            node.value = random.choice([i for i in operands if i != node.value])
        elif node.type == 'constant':
            node.value = random.choice([i for i in constants if i != node.value])
        elif node.type == 'unary_operator':
            node.value = random.choice([i for i in unary_operators if i != node.value])
        elif node.type == 'binary_operator':
            node.value = random.choice([i for i in binary_operators if i != node.value])

    return expression


# def mutate(expression, max_depth, mutation_probability, unary_operators, binary_operators, operands, constants):
#     expression = expression.copy()
#     node, depth = random.choice(get_tree_nodes(expression))
#
#     if random.random() < mutation_probability:
#         new_node = generate_random_tree(max_depth - depth, unary_operators, binary_operators, operands, constants)
#
#         node.value = new_node.value
#         node.left = new_node.left
#         node.right = new_node.right
#         node.type = new_node.type
#
#     return expression


def get_tree_nodes_with_depth(node, target, depth=0):
    if depth == target:
        return [(node, depth)]

    nodes = []

    if node.left is not None:
        nodes += get_tree_nodes(node.left, depth + 1)
    if node.right is not None:
        nodes += get_tree_nodes(node.right, depth + 1)

    return nodes


def get_tree_nodes(node, depth=0, need_root=True):
    if depth == 0 and not need_root:
        nodes = []
    else:
        nodes = [(node, depth)]

    if node.left is not None:
        nodes += get_tree_nodes(node.left, depth + 1)
    if node.right is not None:
        nodes += get_tree_nodes(node.right, depth + 1)

    return nodes


def symbolic_regression(x_train_mappings, y_train, max_depth, population_size, max_generations, selection_size, mutation_probability,
                        unary_operators, binary_operators, operands, constants):
    population = initialize_population(max_depth, population_size, unary_operators, binary_operators, operands, constants)
    fitness = [fitness_function(expression, x_train_mappings, y_train) for expression in population]

    best_expressions = []
    best_expression = None
    best_fitness = None
    counter = 0

    for generation in range(max_generations):
        gc.collect()
        parents, fitness = selection(population, fitness, selection_size)
        if best_fitness == fitness[0]:
            counter += 1
        if counter == 100:
            population = initialize_population(max_depth, population_size, unary_operators, binary_operators, operands, constants)
            fitness = [fitness_function(expression, x_train_mappings, y_train) for expression in population]
            counter = 0
            continue
        best_expression = parents[0]
        best_fitness = fitness[0]
        best_expressions.append(best_expression)

        print("Generation:", generation, "Best fitness:", best_fitness, "Average fitness:", np.mean(fitness))

        children = []

        for i in range(len(parents)):
            parent1 = parents[i]
            parent2 = random.choice(parents)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, max_depth, mutation_probability, unary_operators, binary_operators, operands, constants)
            child2 = mutate(child2, max_depth, mutation_probability, unary_operators, binary_operators, operands, constants)

            children.append(child1)
            children.append(child2)

        population = []

        for child in children + parents:
            if child not in population:
                population.append(child)

        if len(population) < population_size:
            while len(population) < population_size:
                population.append(generate_random_tree(max_depth, unary_operators, binary_operators, operands, constants))

        fitness = [fitness_function(expression, x_train_mappings, y_train) for expression in population]

        if fitness[0] <= 0.00001:
            break

    ind = np.argmin([fitness_function(expression, x_train_mappings, y_train) for expression in best_expressions])
    return best_expressions[ind], best_expressions


def target_function(x):
    return x[:, 0] * np.log(x[:, 1]) - np.exp(x[:, 2])


x_train = np.sort(np.random.uniform(1, 5, size=(50, 3)))
y_train = target_function(x_train)

max_depth = 3
population_size = 20
selection_size = 10
max_generations = 1000
mutation_probability = 1

operands = ['x1', 'x2', 'x3']
# constants = [np.pi]
constants = []
unary_operators = {np.exp: 'exp', minus: '-', np.log: 'ln'}
binary_operators = {np.add: '+', np.multiply: '*', np.power: '^'}

Node.mapping = {**unary_operators, **binary_operators}

x_train_mappings = [{operands[j]: x_train[i, j] for j in range(x_train.shape[1])} for i in range(x_train.shape[0])]

best_individual, best_individuals = symbolic_regression(x_train_mappings, y_train, max_depth, population_size, max_generations, selection_size, mutation_probability,
                                                        list(unary_operators.keys()), list(binary_operators.keys()), operands, constants)

print("Best individual:", best_individual)
print("Fitness:", fitness_function(best_individual, x_train_mappings, y_train))

# x_test = np.sort(np.random.uniform(-np.pi, np.pi, size=(100, 3)))
# y_test = target_function(x_test)
#
# print("Mean squared error test:", -fitness_function(best_individual, x_test_mapping, y_test))
