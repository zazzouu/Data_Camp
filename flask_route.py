from flask import Flask, render_template, request, jsonify, redirect, url_for,  flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, world! This is a simple Flask application.'

if __name__ == '__main__':
    app.run(debug=True)