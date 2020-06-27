from sklearn import metrics, model_selection
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from fizzBuzzDataPrep import *



def predict_range_number(filename, num_from, num_to):
    if (num_from < num_to) and (num_from >= 1 and num_to <=100):
        load_model_lr = joblib.load(filename)
        # select the range [1:100] Ground Truth
        df = pd.read_csv('fist100FizzBuzz_ground_truth.csv')
        y_target = np.array(df['Class'])[num_from-1:num_to]
        preds = []
        result = ""
        for i in range(num_from, num_to+1):
            predict = load_model_lr.predict([factors_prime_encode(i)])
            preds.append(predict[0])
            if switch_fizz_buzz(predict[0]) == "None":
                result += str(i) + " "
            else:
                result += switch_fizz_buzz(predict[0]) + " "

        return result, f1_score(y_target, preds, average='micro')


def train_evaluate_model(filename):
    df = pd.read_csv(filename)
    print("example of first samples")
    print(df.head())
    print("-----------------------------------------")
    print(df.groupby('Class').size())

    data = np.array(df.drop(['Class'], 1))
    target = np.array(df['Class'])
    print("------------ number of samples generated ----------------")

    model_lr = create_model()

    validation_size = 0.20
    seed = 7
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(data, target,
                                                                                    test_size=validation_size,
                                                                                    random_state=seed,
                                                                                    stratify=target)
    # name = 'Logistic Regression'
    kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    model_lr.fit(x_train, y_train)
    cross_validation_scores = model_selection.cross_val_score(model_lr, x_train, y_train, cv=kfold, scoring="f1_micro")

    preds = model_lr.predict(x_validation)
    classif_name = 'classifier_' + str(data.shape[0]) + '_data_sample' + '.pkl'
    save_object(classif_name, model_lr)

    print("------------ classification report ----------------")
    print(classification_report(y_validation,preds))
    return cross_validation_scores, \
           cross_validation_scores.mean(), \
           accuracy_score(y_validation, preds),\
           precision_score(y_validation, preds, average='micro'), \
           recall_score(y_validation, preds, average='micro'), \
           f1_score(y_validation, preds, average='micro'), \
           classif_name

