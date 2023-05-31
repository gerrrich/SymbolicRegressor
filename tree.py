import math
import random

import numpy as np


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

    for i in range(population_size):
        tree = generate_random_tree(max_depth, unary_operators, binary_operators, operands, constants)
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


class Node:
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


def selection(population, fitness, selection_size):
    best_indices = np.argsort(fitness)[:selection_size]
    return [population[i] for i in best_indices], [fitness[i] for i in best_indices]


def crossover(parent1, parent2):
    if parent1.type in ['operand', 'constant'] or parent2.type in ['operand', 'constant']:
        return parent2.copy(), parent1.copy()

    nodes1 = get_tree_nodes(parent1.copy(), need_root=False)
    node1, depth1 = random.choice(nodes1)

    nodes2 = get_tree_nodes_with_depth(parent2.copy(), target=depth1)
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

    return parent1, parent2


def mutate(expression, max_depth, mutation_probability, unary_operators, binary_operators, operands, constants):
    node = random.choice(get_tree_nodes(expression))[0]

    if node.type == 'operand':
        node.value = random.choice(operands)
    elif node.type == 'constant':
        node.value = random.choice(constants)
    elif node.type == 'unary_operator':
        node.value = random.choice(unary_operators)
    elif node.type == 'binary_operator':
        node.value = random.choice(binary_operators)

    return expression


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

    for generation in range(max_generations):
        parents, fitness = selection(population, fitness, selection_size)

        best_expression = parents[0]
        best_expressions.append(best_expression)

        print("Generation:", generation, "Best fitness:", fitness[0])

        children = []

        for i in range(len(parents)):
            parent1 = parents[i]
            parent2 = random.choice(parents)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, max_depth, mutation_probability, unary_operators, binary_operators, operands, constants)
            child2 = mutate(child2, max_depth, mutation_probability, unary_operators, binary_operators, operands, constants)

            children.append(child1)
            children.append(child2)

        population = parents + children
        fitness = [fitness_function(expression, x_train_mappings, y_train) for expression in population]

        # best_indices = np.argsort(fitness)[:population_size]
        # population = [population[i] for i in best_indices]
        # fitness = [fitness[i] for i in best_indices]

    return best_expression, best_expressions


def target_function(x):
    return np.sin(x[:, 0]) * np.cos(x[:, 1]) + np.exp(x[:, 2])


x_train = np.sort(np.random.uniform(-np.pi, np.pi, size=(100, 3)))
y_train = target_function(x_train)


def minus(x):
    return -x


max_depth = 4
population_size = 20
max_generations = 50
selection_size = 10
mutation_probability = 0.1

operands = ['x1', 'x2', 'x3']
constants = [np.pi]
unary_operators = [np.sin, np.cos, minus, np.log10, np.log, np.exp]
binary_operators = [np.add, np.subtract, np.multiply, np.divide, np.power]

x_train_mappings = [{operands[j]: x_train[i, j] for j in range(x_train.shape[1])} for i in range(x_train.shape[0])]

best_individual, best_individuals = symbolic_regression(x_train_mappings, y_train, max_depth, population_size, max_generations, selection_size, mutation_probability,
                                                        unary_operators, binary_operators, operands, constants)

print("Best individual:", best_individual)
print("Fitness:", fitness_function(best_individual, x_train_mappings, y_train))

# x_test = np.sort(np.random.uniform(-np.pi, np.pi, size=(100, 3)))
# y_test = target_function(x_test)
#
# print("Mean squared error test:", -fitness_function(best_individual, x_test_mapping, y_test))
