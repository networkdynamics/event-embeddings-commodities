import torch
import numpy as np

def test_sampler():
    cum_sizes = [20, 26]
    all_cum_lengths = [[10, 20], [4, 6]]

    dataset_indices = []
    for dataset_idx in range(len(all_cum_lengths)):
        cum_lengths = all_cum_lengths[dataset_idx]
        chunk_order = torch.randperm(len(cum_lengths))
        chunks = [0] + cum_lengths
        dataset_indices.append([])
        for chunk_idx in chunk_order:
            dataset_cum_size = 0 if dataset_idx == 0 else cum_sizes[dataset_idx - 1]
            indices = torch.randperm(chunks[chunk_idx + 1] - chunks[chunk_idx]) + chunks[chunk_idx] + dataset_cum_size
            dataset_indices[dataset_idx] += indices.tolist()

    indices_options = list(range(len(dataset_indices)))
    indices_idx = [0] * len(dataset_indices)
    indices_left = np.array([len(indices) for indices in dataset_indices])
    indices = []
    while sum(indices_left) > 0:
        idx = np.random.choice(indices_options, p=indices_left / np.sum(indices_left))

        indices.append(dataset_indices[idx][indices_idx[idx]])
        indices_idx[idx] += 1
        indices_left[idx] -= 1

    assert len(indices) == sum([len(set_indices) for set_indices in dataset_indices])
    assert len(indices) == cum_sizes[-1]
    for num in range(cum_sizes[-1]):
        assert num in indices

    print(indices)


def main():
    test_sampler()

if __name__ == '__main__':
    main()
