import joblib
from sklearn import linear_model
from sympy.ntheory import primefactors as pf

def switch_index_encode(argument):
    switcher = {
        2: 0,
        3: 1,
        5: 2,
        7: 3,
        11: 4,
        13: 5
    }
    return (switcher.get(argument, -1))

def factors_prime_encode(number):
    factors_x = pf(number)
    code = [0] * 6
    for factor in factors_x:
        index = switch_index_encode(factor)
        if index != -1:
            code[index] = 1
    return code


def fizzbuzz(i):
    if i % 15 == 0:
        return 1
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 3
    else:
        return 4

def switch_fizz_buzz(argument):
    switcher = {
        4: "None",
        3: "Fizz",
        2: "Buzz",
        1: "FizzBuzz"
    }
    return (switcher.get(argument, -1))

def save_object(filename, model):
    with open(''+filename, 'wb') as file:
        joblib.dump(model, filename)


def load_object(filename):
    with open(''+filename ,'rb') as f:
        loaded = joblib.load(f)
    return loaded

def create_model():
    model_lR = linear_model.LogisticRegression(
        C=1.0, class_weight=None, dual=False,
        fit_intercept=True, intercept_scaling=1, max_iter=1000,
        multi_class='ovr',
        n_jobs=1, penalty='l2',
        random_state=None,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)
    return model_lR

