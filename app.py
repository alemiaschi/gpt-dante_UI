from flask import Flask, render_template, request
from generator_script import generate

app = Flask(__name__)

@app.route("/")
def my_form():
    return render_template('index.html')

@app.route("/", methods=['GET','POST'])

def func_test():
    if request.method == 'POST':
        prompt = request.form['myTextArea']
        output = generate(prompt)
        return render_template('index.html', output_area=output)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
