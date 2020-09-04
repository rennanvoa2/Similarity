import pandas as pd
import numpy as np
from flask import Flask, render_template
from flask import request
import spacy


app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

@app.route("/similarity")
def selected_skills_test():
    string1 = request.args.get('string1')
    string2 = request.args.get('string2')

    doc1 = nlp(string1)
    doc2 = nlp(string2)

    similarity = doc1.similarity(doc2)

    return {"similarity":similarity}


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5005,debug=True)
    app.run(host='127.0.0.1', port=5005, debug=True)

