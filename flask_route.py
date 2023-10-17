from flask import Flask, render_template, request, jsonify, redirect, url_for,  flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('page.html')

if __name__ == '__main__':
    app.run(debug=True)