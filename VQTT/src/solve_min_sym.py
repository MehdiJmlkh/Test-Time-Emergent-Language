from itertools import combinations, chain, product
import random
from collections import defaultdict
from torch.utils.data import random_split
from functools import reduce
import operator


def SolveMinSym(target_image, all_images):
    """
    Minimum number of positions required to uniquely
    identify the target image.
    """
    distracting_images = [
        img for img in all_images if img != target_image
    ]

    for combination in attribute_combinations(target_image):
        if is_unique_combination(combination, target_image, distracting_images):
            return len(combination)

    return None


def attribute_combinations(image):
    """
    Generate combinations of attribute INDICES
    (not values).
    """
    indices = list(range(len(image)))

    return chain.from_iterable(
        combinations(indices, r)
        for r in range(1, len(indices) + 1)
    )


def is_unique_combination(combination, target_image, distracting_images):
    """
    Check if selected positions uniquely identify target.
    """
    for image in distracting_images:
        match = all(
            image[idx] == target_image[idx]
            for idx in combination
        )

        if match:
            return False

    return True


def min_m_controlled_sampling(dataset, number_of_samples, target_min_symbol, batch_size):

    result = []
    while (len(result) * batch_size) < number_of_samples:

        batch = random.sample(dataset, batch_size)
        target = random.sample(batch, 1)[0]
        
        batch_indices = [image[1] for image in batch]
        batch_values = [image[0] for image in batch]
        target_value, target_index = target
        
        min_symbol = SolveMinSym(target_value, batch_values)
        
        if min_symbol == target_min_symbol:
            result.append({target_index: batch_indices})

    return reduce(operator.or_, result)