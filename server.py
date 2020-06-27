from flask import Flask, request, make_response
from fizzBuzzLogReg import *

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route("/", methods=['GET'])
def front():
    return '''<form action="/" method="GET" >
            <h1>Generate data training: {}</h1>
            <h1>train and evaluate model: {}</h1>
            <h1>Predict range of number: {}</h1>'''.format("/generate", "/train", "/predict")


@app.route("/generate", methods=["GET", "POST"])
def generate_data():
    if request.method == "POST":
        sample_size = request.form.get("Sample Size")
        result = generate_data_sample(sample_size)
        resp = make_response(result.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=sample_data.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    else:
        return '''<form action="/generate" method="POST" enctype=multipart/form-data>
                <label>Sample size:</label>
                <input type="text" name="Sample Size"/>                
                <input type="submit" value="Send"></form>'''


@app.route("/train", methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        f = request.files['archive']
        cross_validation_scores, cross_validation_score_mean, accuracy_score, precision_score, recall_score, f1_score, classifer_name = \
            train_evaluate_model(f.filename)
        return '''<h1>Cross_validation metrics: {}</h1>
                    <h1>Cross_validation metric mean: {}</h1>
                    <h1>Accuracy score: {}</h1>
                    <h1>Precision score: {}</h1>
                    <h1>Recall score: {}</h1>
                    <h1>F1 Score: {}</h1>
                    <h1>Save model: {}</h1>''' \
            .format(cross_validation_scores, cross_validation_score_mean, accuracy_score, precision_score, recall_score,
                    f1_score, classifer_name)
    else:
        return '''<form action="/train" method="POST" enctype="multipart/form-data">       
                <label>Select data sample to train model:</label>
                 <input type="file" name="archive">
                 <input type="submit" value="Send" /></form>'''


@app.route("/predict", methods=["GET", "POST"])
def predict_range():
    if request.method == "POST":
        file_classifer = request.files['archive']
        numFrom = request.form.get("from")
        numTo = request.form.get("to")
        result, f1_score = predict_range_number(file_classifer.filename, int(numFrom), int(numTo))
        return '''<h1>Fizz Buzz in range {}</h1>
               <h1>F1 Score: {}</h1>'''.format("[" + str(numFrom) + " - " + str(numTo) + "]:   " + result,
                                               str(f1_score))
    else:
        return '''<form action="/predict" method="POST" enctype=multipart/form-data>
                <label>Range of numbers to classifer</label>
                <label>from:</label>
                <input type="text" name="from"/>
                <label>to:</label>
                <input type="text" name="to"/><br/><br/>
                <label>Select classifer .pkl:</label>
                <input type=file name=archive>
                <input type=submit value="Send"></form>'''


if __name__ == '__main__':
    print("Generate Ground Truth to first 100 numbers")
    generate_first100_fizz_buzz()
    app.run()
