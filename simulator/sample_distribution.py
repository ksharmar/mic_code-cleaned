import numpy as np
# np.random.seed(31)
import random
# random.seed(0)

def sample_from_power_law(alpha, min_val, max_val, num_samples=1):
    """
    Sampling from the power law distribution

    Parameters
    ----------
    alpha: float
    min_val: float
    max_val: float
    num_samples: int (optional) default=1

    Returns
    -------
    float
        num_samples sampled from the distribution
        (power-law) with pdf(x) prop to x^{-alpha-1} for min<=x<=max.
    """
    exp = -alpha
    r = np.random.random(size=num_samples)
    ag, bg = min_val**exp, max_val**exp
    return (ag + (bg - ag)*r)**(1./exp)


def sample_seed_sets(users, alpha, num_seedsets):
    # sample seed set sizes from power law
    ss_sizes = sample_from_power_law(alpha, 1, len(users), num_seedsets)
    users = np.array(users, dtype=np.int64)
    seed_sets = [np.random.choice(list(users), int(size), replace=False) for size in ss_sizes]
    return seed_sets


def replicate_seed_sets(unique_seed_sets, low, high):
    seed_sets = list()
    for unique in unique_seed_sets:
        num_cascades = np.random.randint(low, high)
        repeated = np.repeat([unique], repeats=num_cascades, axis=0)
        seed_sets += list(repeated)
    random.Random(4).shuffle(seed_sets)
    return np.array(seed_sets)