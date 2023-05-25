import random
import operator
import math


# Определяем функцию для создания случайного символьного выражения
def generate_expression(depth):
    if depth == 0:
        return random.choice(variables)

    task = random.choice(operators + functions)

    if task in operators:
        left = generate_expression(depth - 1)
        right = generate_expression(depth - 1)
        return task(left, right)

    else:
        value = generate_expression(depth - 1)
        return task(value)


# Определяем функцию для эволюции символьных выражений с использованием генетического программирования
def evolve(target, population_size, mutation_rate, max_depth, max_generations, x, y, z, error, dead_rate):
    population = [generate_expression(random.randint(1, max_depth)) for _ in range(population_size)]  # max_depth // 2

    for generation in range(max_generations):
        scores = [(evaluate(expression, x, y, z), expression, abs(evaluate(expression, x, y, z) - target)) for expression in population]
        best_score = min(scores, key=lambda r: r[2])

        if best_score[2] <= error:
            return best_score

        print(f'Generation {generation}: best score = {best_score[0]}, min error = {best_score[2]}')

        random.shuffle(scores)
        deads_count = int(population_size * dead_rate)
        deads_count = deads_count - 1 if deads_count % 2 == 0 else deads_count
        deads = scores[:deads_count]
        born = [best_score]
        alive = scores[deads_count:]

        for i in range(1, deads_count, 2):
            parent1 = deads[i]
            parent2 = deads[i + 1]
            child1, child2 = crossover(parent1, parent2)

            if random.random() < mutation_rate:
                child1 = mutate(child1, max_depth)

            if random.random() < mutation_rate:
                child2 = mutate(child2, max_depth)

            born.append(child1)
            born.append(child2)

        population = [i[1] for i in born + alive]

    return best_score


# Определяем функцию для выбора родителя
# def select(scores):
#     return random.choice(scores)[1]
# total = sum(score[0] for score in scores)
# r = random.uniform(0, total)
# s = 0
#
# for score in scores:
#     s += score[0]
#     if s >= r:
#         return score[1]


# Определяем функцию для скрещивания
def crossover(parent1, parent2):
    # index = random.randint(0, min(len(parent1), len(parent2)) - 1)
    # return parent1[:index] + parent2[index:]
    return parent2, parent1


# Определяем функцию для мутации
def mutate(child, max_depth):
    variables_to_chuse = variables.copy()

    while True:
        rand_var = random.choice(variables_to_chuse)

        if child[1].find(rand_var) != -1:
            break
        else:
            variables_to_chuse.remove(rand_var)

            if len(variables_to_chuse) == 0:
                return child

    index = random.choice([i for i in range(len(child[1])) if child[1][i] == rand_var])

    return child[1][:index] + generate_expression(1) + child[1][index + 1:]

    # if child[index] in variables:
    #     return child[:index] + generate_expression(1) + child[index + 1:]
    # elif len(child) >= max_depth:
    #     return child
    # else:
    #     return child[:index] + generate_expression(1) + child[index:]


# Пример использования функции evolve для поиска символьного выражения, которое наилучшим образом соответствует некоторому целевому значению
x = 20
y = 30
z = 50

target = (math.sin(x) / math.cos(y)) / math.cos(z)

expression = evolve(target, 1000, 0.9, 4, 400, x, y, z, 0.0001, 0.5)
print(f'Best expression for target value {target}: {expression[0]} = {expression[1]}, error = {expression[2]}')
