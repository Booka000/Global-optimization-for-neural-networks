import numpy as np
from scipy.stats import cauchy
from tqdm import tqdm


def truncated_cauchy(mean, std, upper=1.0):
    x = cauchy.rvs(mean, std)
    if x > upper:
        x = upper
        return x
    elif x < 0:
        return truncated_cauchy(mean, std, upper)
    else:
        return x


def lehmer_mean(lst):
    if len(lst) == 0:
        return 0

    numerator = np.sum(np.power(lst, 2))
    denominator = np.sum(lst)
    return numerator / denominator


def inverse_probability(epsilon, x_0, x_min, x_max):
    y = np.random.random(len(x_min))
    a = (x_max - x_0) / epsilon
    b = (x_min - x_0) / epsilon
    return epsilon * np.tan(y * np.arctan(a) + (1.0 - y) * np.arctan(b)) + x_0


def SoFA_mo(fitFunction, boundaries, initial_population_size=100, max_iter=10000, mu_mean=0.5,
         c=0.5, eps_len=300, eps_h_bound=1.2, disable_pbar=False):
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
    eps_mean = mu_mean
    counter = 0
    gen_counter = 0
    S_epsilon = []
    loss_history = list()
    pbar = tqdm(total=max_iter, bar_format='{percentage:3.0f}%|{bar:75}{r_bar}{n} {desc:50}',
                desc="best of initial population: %f" % fittestValue, disable=disable_pbar)
    while counter < max_iter:
        counter += 1
        epsilon = truncated_cauchy(eps_mean, eps_h_bound)
        index = np.random.choice(len(points), p=probabilities)
        newPoint = inverse_probability(epsilon, points[index], boundaries[:, 0], boundaries[:, 1])
        newFitness = fitFunction(newPoint)
        if newFitness > fitnesses[index]:
            S_epsilon.append(epsilon)
        points.append(newPoint)
        fitnesses = np.insert(fitnesses, len(fitnesses), newFitness)
        if newFitness > fittestValue:
            fittestPoint = newPoint
            fittestValue = newFitness
            pbar.set_description("iteration %d : %f epsilon = %f" %
                                 (counter, np.absolute(fittestValue), epsilon), refresh=True)
        elif newFitness < worstValue:
            worstValue = newFitness
        loss_history.append(fittestValue)
        width = fittestValue - worstValue
        numerator = ((fitnesses - worstValue) / width) ** len(fitnesses)
        denominator = np.sum(numerator)
        probabilities = numerator / denominator
        probabilities = np.absolute(probabilities)
        if gen_counter == eps_len - 1:
            eps_mean = (1 - c) * eps_mean + c * lehmer_mean(S_epsilon)
            S_epsilon = []
            gen_counter = 0
        else:
            gen_counter += 1
        pbar.update(1)
    pbar.close()
    return [fittestPoint, fittestValue, loss_history]

#%%
