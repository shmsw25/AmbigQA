import json
import numpy as np
from flask import Flask, render_template, redirect, request, jsonify, make_response

from Interactive import InteractiveDPR

app = Flask(__name__)
debug = False

if not debug:
    dpr = InteractiveDPR()

with open("data/nqopen-dev.json", "r") as f:
    data = json.load(f)
questions = [d["question"] for d in data]

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    indices = np.random.permutation(range(len(questions)))[:100]
    return jsonify(result={"questions": [questions[i] for i in indices]})

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    question = request.args.get('question')
    k = request.args.get('k')
    k = int(k)
    if debug:
        result = [{
            "title": "title-{}".format(i),
            "passage_index": i,
            "text": "answer",
            "passage": "random text with answer",
            "softmax": {"span": 0.3, "passage": 0.5, "joint": 0.15},
        } for i in range(k)]
    else:
        result = dpr.run(question, topk_answer=k)

    return jsonify(result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2020, threaded=True)







