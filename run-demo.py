import json
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, redirect, request, jsonify, make_response


app = Flask(__name__)

@app.route('/generation')
def main_generation():
    return render_template('generation.html')

@app.route('/validation')
def main_validation():
    return render_template('validation.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    return jsonify(result={"contextss" : contextss,
        "titles": titles,
        "context_questions": context_questions,
        "all_contextss": all_contextss,
        'all_questions': all_questions})

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    paragraphs = request.args.get('paragraphs')
    question = request.args.get('question')
    reasoningType = int(request.args.get('reasoningType'))

    if get_key(paragraphs, question, reasoningType) in cached_output:
        answer = cached_output[get_key(paragraphs, question, reasoningType)]
    else:
        answer = get_answer([p for p in paragraphs.split('\n') if len(p.strip())>0],
                        question,
                        reasoningType)
    return jsonify(result=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1999, threaded=True)







