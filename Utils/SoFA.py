import numpy as np
from tqdm import tqdm
import gc

def inverse_probability(epsilon, x_0, x_min, x_max):
    y = np.random.random(len(x_min))
    a = (x_max - x_0) / epsilon
    b = (x_min - x_0) / epsilon
    return epsilon * np.tan(y * np.arctan(a) + (1.0 - y) * np.arctan(b)) + x_0


def __dispersion(length, initial_population_size, dispersion_a, dispersion_b):
    kk = length - initial_population_size
    if kk == 0:
        kk = 1
    return kk ** (-dispersion_a - dispersion_b * kk)


def SoFA(fitFunction, boundaries, scbd=0.01, initial_population_size=100,
         max_iter=10000, disable_pbar=False):
    points = [np.random.uniform(boundaries[:, 0], boundaries[:, 1]) for _ in range(initial_population_size)]
    fitnesses = np.asarray([fitFunction(_) for _ in points])
    fittestValue = np.max(fitnesses)
    worstValue = np.min(fitnesses)
    fittestPoint = points[np.argmax(fitnesses)]
    width = fittestValue - worstValue
    numerator = ((fitnesses - worstValue) / width) ** len(fitnesses)
    denominator = np.sum(numerator)
    probabilities = numerator / denominator
    probabilities = np.absolute(probabilities)
    dispersion_a = 0.4
    dispersion_b = 2.5e-6
    counter = 0
    epsilon_counter = 0
    epsilon = __dispersion(len(points), initial_population_size, dispersion_a, dispersion_b)
    LossHistory = list()
    pbar = tqdm(total=max_iter, bar_format='{percentage:3.0f}%|{bar:75}{r_bar}{n} {desc:50}',
                desc="best of initial population: %f" % fittestValue, disable=disable_pbar)
    while epsilon >= scbd and counter < max_iter:
        counter += 1
        index = np.random.choice(len(points), p=probabilities)
        newPoint = inverse_probability(epsilon, points[index], boundaries[:, 0], boundaries[:, 1])
        newFitness = fitFunction(newPoint)
        points.append(newPoint)
        fitnesses = np.insert(fitnesses, len(fitnesses), newFitness)
        if newFitness > fittestValue:
            fittestPoint = newPoint
            fittestValue = newFitness
            pbar.set_description("iteration %d : %f epsilon = %f epsilon_counter - %d" %
                                 (counter, np.absolute(fittestValue), epsilon, epsilon_counter), refresh=True)
            epsilon_counter -= 1
        elif newFitness < worstValue:
            worstValue = newFitness
            epsilon_counter += 1
        else:
            epsilon_counter += 1
        LossHistory.append(fittestValue)
        width = fittestValue - worstValue
        numerator = ((fitnesses - worstValue) / width) ** len(fitnesses)
        denominator = np.sum(numerator)
        probabilities = numerator / denominator
        probabilities = np.absolute(probabilities)
        if epsilon < 0.01:
            epsilon = __dispersion(len(points) - epsilon_counter, initial_population_size, dispersion_a, dispersion_b)
        else:
            epsilon = __dispersion(len(points), initial_population_size, dispersion_a, dispersion_b)
        pbar.update(1)
    pbar.close()
    del points
    del probabilities
    del fitnesses
    gc.collect()
    return [fittestPoint, fittestValue, LossHistory]
