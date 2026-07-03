import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import numpy as np
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import torch
import math
from collections import Counter


def get_messages(
    sender, test_dataset, device, message_length, batch_size=100, num_workers=0
):
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_messages = []

    sender.eval()

    with torch.no_grad():
        for iter_num, (imgs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs = imgs.to(device)

            # Generate messages from images using agent A
            generated_output = sender.forward_text_generation(
                imgs,
                message_length=message_length,
                freeze_codebook=True,
                mode="discrete",
                sampling_temperature=1e-5,
            )

            # words: [batch_size, message_length]
            words = generated_output["indices"]

            for msg in words.cpu().tolist():
                all_messages.append(tuple(msg))

    sender.train()

    return all_messages


def count_unique_messages(messages):
    return len({tuple(list(m)) for m in messages})


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def topographic_similarity(meanings, messages):
    meanings = list(meanings)
    messages = list(messages)

    meaning_distances = []
    message_distances = []

    for i, j in combinations(range(len(meanings)), 2):
        meaning_distances.append(hamming_distance(meanings[i], meanings[j]))
        message_distances.append(hamming_distance(messages[i], messages[j]))

    rho, _ = spearmanr(meaning_distances, message_distances)
    return float(rho)


def adjusted_mutual_information(messages, meanings):
    msg_ids = [tuple(m.tolist()) for m in messages]
    meaning_ids = [tuple(z.tolist()) for z in meanings]

    msg_labels = {m: i for i, m in enumerate(set(msg_ids))}
    meaning_labels = {z: i for i, z in enumerate(set(meaning_ids))}

    msg_enc = [msg_labels[m] for m in msg_ids]
    meaning_enc = [meaning_labels[z] for z in meaning_ids]

    return adjusted_mutual_info_score(msg_enc, meaning_enc)


def positional_disentanglement(messages, meanings):
    N, L = messages.shape
    A = meanings.shape[1]

    scores = []

    for i in range(L):
        mi = []
        for a in range(A):
            mi.append(mutual_info_score(messages[:, i], meanings[:, a]))

        total = sum(mi)
        if total > 0:
            scores.append(max(mi) / total)

    return float(np.mean(scores))


def bag_of_symbols_disentanglement(messages, meanings):
    vocab = torch.unique(messages)
    A = meanings.shape[1]

    scores = []

    for w in vocab:
        presence = (messages == w).any(dim=1).long()

        mi = []
        for a in range(A):
            mi.append(mutual_info_score(presence, meanings[:, a]))

        total = sum(mi)
        if total > 0:
            scores.append(max(mi) / total)

    return float(np.mean(scores))


def context_independence(messages, meanings):
    vocab = torch.unique(messages)
    N, A = meanings.shape

    scores = []

    for w in vocab:
        mask = (messages == w).any(dim=1)
        if mask.sum() == 0:
            continue

        probs = []
        for a in range(A):
            values = torch.unique(meanings[:, a])
            for v in values:
                p = (meanings[mask, a] == v).float().mean()
                probs.append(p)

        total = sum(probs)
        if total > 0:
            scores.append(max(probs) / total)

    return float(np.mean(scores))


def concept_best_matching(messages, meanings):
    vocab = torch.unique(messages)
    N, A = meanings.shape

    cbm_scores = []

    for a in range(A):
        values = torch.unique(meanings[:, a])

        for v in values:
            mask = meanings[:, a] == v
            if mask.sum() == 0:
                continue

            freqs = []
            for w in vocab:
                freq = (messages[mask] == w).float().mean()
                freqs.append(freq)

            cbm_scores.append(max(freqs))

    return float(torch.stack(cbm_scores).mean())


def encode_as_ids(tensor):
    """
    tensor: (N, D) discrete tensor
    returns: list of tuples
    """
    return [tuple(row.tolist()) for row in tensor]


def estimate_distributions(messages, meanings):
    """
    messages: (N, L)
    meanings: (N, C)

    returns:
        P_cm: dict (c, m) -> P(c,m)
        P_c:  dict c -> P(c)
        P_m:  dict m -> P(m)
    """
    msg_ids = encode_as_ids(messages)
    meaning_ids = encode_as_ids(meanings)

    joint_counts = Counter(zip(meaning_ids, msg_ids))
    N = sum(joint_counts.values())

    # Joint probability
    P_cm = {k: v / N for k, v in joint_counts.items()}

    # Marginals
    P_c = Counter()
    P_m = Counter()

    for (c, m), p in P_cm.items():
        P_c[c] += p
        P_m[m] += p

    return P_cm, P_c, P_m


def conditional_entropy(P_joint, P_given):
    """
    Computes H(X | Y)

    P_joint: dict (x, y) -> P(x,y)
    P_given: dict y -> P(y)
    """
    H = 0.0
    eps = 1e-12

    for (x, y), p_xy in P_joint.items():
        p_y = P_given[y]
        p_x_given_y = p_xy / p_y
        H -= p_xy * math.log(p_x_given_y + eps)

    return H


def compute_conditional_entropies(messages, meanings):
    """
    messages: (N, L) discrete
    meanings: (N, C) discrete

    returns:
        H_C_given_M
        H_M_given_C
    """
    P_cm, P_c, P_m = estimate_distributions(messages, meanings)

    # H(C | M)
    P_c_m = {(c, m): p for (c, m), p in P_cm.items()}
    H_C_given_M = conditional_entropy(P_c_m, P_m)

    # H(M | C)
    P_m_c = {(m, c): p for (c, m), p in P_cm.items()}
    H_M_given_C = conditional_entropy(P_m_c, P_c)

    return H_C_given_M, H_M_given_C
