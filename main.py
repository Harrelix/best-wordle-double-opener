import pickle
from itertools import combinations, combinations_with_replacement, permutations, islice
from typing import List, Generator, Literal
import numpy as np
from numpy import log2, random
from numpy.typing import NDArray

# params
BUFFER = 100
N = 4000

# colors
gray: Literal[0] = 0
green: Literal[1] = 1
yellow: Literal[2] = 2


def is_possible_answer(answer: str, word: str, info: List[Literal[0, 1, 2]]) -> bool:
    """check if `answer` is a possible answer, given the `word`, and the `infos` (list of colors)"""
    for i in range(len(word))[::-1]:
        if word[i] == " ":
            continue
        if info[i] == green:
            if answer[i] != word[i]:
                return False
            answer = answer.replace(word[i], "-", 1)
        elif info[i] == yellow:
            if answer[i] == word[i]:
                return False
            if not word[i] in answer:
                return False
            answer = answer.replace(word[i], "-", 1)
        elif info[i] == gray:
            if word[i] in answer:
                return False
    return True

class SymmetricMatrix:
    def __init__(self, size, dtype):
        self._size = size
        self._data = np.zeros(size * (size - 1) // 2, dtype)

    def __len__(self):
        return self._size

    def __setitem__(self, position, value):
        index = self._get_index(position)
        self._data[index] = value

    def __getitem__(self, position):
        index = self._get_index(position)
        return self._data[index]

    def _get_index(self, position):
        row, column = position
        if column == row:
            raise IndexError("column == row")
        if column < row:
            row, column = column, row
        index = (column - 1) + row * (self._size - 1) - row * (row + 1) // 2
        return index

    def load(data):
        m = SymmetricMatrix(0, data.dtype)
        m._data = data
        return m

    def get_data(self):
        return self._data


class SymmetricDoubleKeySortedMap:
    def __init__(self):
        self.keys = []
        self.vals = {}

    def add(self, key, val):
        n1, n2 = key
        if n2 > n1:
            n2, n1 = n1, n2

        if (n1, n2) in self.vals:
            self.keys.remove((n1, n2))
        self.vals[n1, n2] = val
        if val > self.vals[self.keys[0]]:
            self.keys.insert(0, (n1, n2))
        elif val < self.vals[self.keys[-1]]:
            self.keys.append((n1, n2))
        else:
            self.keys.insert(self._binary_search(0, len(self.keys) - 1, val), (n1, n2))

    def _binary_search(self, start, end, val):
        if start + 1 == end:
            return end
        mid = (start + end) // 2
        if self.vals[self.keys[mid]] < val:
            return self._binary_search(start, mid, val)
        else:
            return self._binary_search(mid, end, val)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, position):
        if type(position) == tuple:
            n1, n2 = position
            if n2 > n1:
                return self.vals[n2, n1]
            return self.vals[n1, n2]
        elif type(position) == int:
            return self.vals[self.keys[position]]

    def get_largest_pair(self):
        return (self.keys[0], self.vals[self.keys[0]])

    def __iter__(self):
        return ((k, self.vals[k]) for k in self.keys)


def load_answers_left() -> dict[Literal[0, 1, 2], dict[str, List[set[int]]]]:
    with open("possible_answers_left.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def save_calculated(m: SymmetricMatrix):
    with open("calculated.npy", "wb") as f:
        np.save(f, m.get_data())


def load_calculated() -> SymmetricMatrix:
    with open("calculated.npy", "rb") as f:
        m = SymmetricMatrix.load(np.load(f))
    return m


def save_double_map(m: SymmetricDoubleKeySortedMap):
    with open("double_map.pkl", "wb") as f:
        pickle.dump(m, f)


def load_double_map() -> SymmetricDoubleKeySortedMap:
    with open("double_map.pkl", "rb") as f:
        m = pickle.load(f)
    return m


def all_possible_result_generator() -> Generator[
    tuple[Literal[0, 1, 2], ...], None, None
]:
    for c in combinations_with_replacement((gray, green, yellow), 5):
        for p in permutations(c):
            yield p


def main():
    # get wordle's allowed words and possible words
    print("loading words...", end="")
    with open("allowed_words.txt", "r") as f:
        allowed_words = f.read().split()
    with open("possible_words.txt", "r") as f:
        possible_words = f.read().split()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    possible_len = len(possible_words)
    allowed_len = len(allowed_words)
    print("done")

    infos = list(set(all_possible_result_generator()))

    # load calculated data
    print("loading calculated data...", end="")
    calculated = load_calculated()
    double_map = load_double_map()
    answers_left_data = load_answers_left()
    print("done")

    def get_entropies() -> NDArray[np.float64]:
        entropies = np.empty(allowed_len)
        for i, w in enumerate(answers_left_data.keys()):
            for k in range(len(answers_left_data[w])):
                p = len(answers_left_data[w][k]) / possible_len
                if p == 0:
                    continue
                entropies[i] -= p * log2(p)
        return entropies

    entropies: NDArray[np.float64] = get_entropies()

    def normalize(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return arr / arr.sum()

    def get_pairs_entropy(n1, n2):
        entropy = 0
        for i1 in range(len(infos)):
            al1 = set(answers_left_data[allowed_words[n1]][i1])
            for i2 in range(len(infos)):
                al2 = set(answers_left_data[allowed_words[n2]][i2])
                num_words_left = len(al1 & al2)
                if num_words_left == 0:
                    continue
                p = num_words_left / possible_len
                entropy -= p * log2(p)
        return entropy

    ne1 = normalize(entropies[:-1])

    top_5_largest_pair = list(islice(iter(double_map), 5))
    while len(top_5_largest_pair) < 5:
        top_5_largest_pair.append(((-1, -1), -1))

    def add_random_entropy():
        while True:
            n1 = random.choice(allowed_len - 1, p=ne1)
            n2 = random.choice(
                np.arange(n1 + 1, allowed_len), p=normalize(entropies[n1 + 1 :])
            )
            if not calculated[n1, n2]:
                break

        entropy = get_pairs_entropy(n1, n2)
        double_map.add((n1, n2), entropy)
        for j, (old_key, old_val), ((nn1, nn2), new_val) in zip(
            range(5), top_5_largest_pair, islice(iter(double_map), 5)
        ):
            if new_val > old_val:
                top_5_largest_pair[j] = ((nn1, nn2), new_val)
                with open("log.txt", "a") as f:
                    f.write(
                        f"{j} {len(double_map)} {allowed_words[nn1]} {allowed_words[nn2]} {new_val}\n"
                    )

        calculated[n1, n2] = True

    for i in range(N):
        add_random_entropy()
        if i % BUFFER == 0 and i != 0:
            save_calculated(calculated)
            save_double_map(double_map)
            print("\r", i, end="")
    save_calculated(calculated)
    save_double_map(double_map)
    print("done")


if __name__ == "__main__":
    main()
