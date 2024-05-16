from flask import Flask,request, render_template

app = Flask(__name__)


#routes
@app.route('/')
def index():
    return render_template('index.html')

#main
if __name__ == '__main__':
    app.run(debug=True)