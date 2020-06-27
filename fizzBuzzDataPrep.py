import pandas as pd
import numpy as np
from usefulMethods import *


def generate_data_sample(sample_size):
    x_data = np.array([factors_prime_encode(i) for i in range(101, 101 + int(sample_size))])
    y_target = np.array([[fizzbuzz(i)] for i in range(101, 101 + int(sample_size))])
    z_samples = np.append(x_data, y_target, axis=1)
    df = pd.DataFrame(data=z_samples, columns=["2", "3", "5", "7", "11", "13", "Class"])
    return df


def generate_first100_fizz_buzz():
    sample_fizz_buzz = np.array([factors_prime_encode(i) for i in range(1, 100 + 1)])
    target_fizz_buzz = np.array([[fizzbuzz(i)] for i in range(1, 100 + 1)])
    z_fizz_buzz = np.append(sample_fizz_buzz, target_fizz_buzz, axis=1)
    df = pd.DataFrame(data=z_fizz_buzz, columns=["2", "3", "5", "7", "11", "13", "Class"])
    df.to_csv('fist100FizzBuzz_ground_truth.csv', index=False)
