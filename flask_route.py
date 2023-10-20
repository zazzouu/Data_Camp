from flask import Flask, render_template, request, jsonify, redirect, url_for,  flash
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Définissez le chemin où vous souhaitez enregistrer les images téléchargées
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/uploads', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            # Assurez-vous que l'extension est .png
            if image.filename.endswith('.png'):
                try:
                    # Renommer le fichier en utilisant secure_filename
                    filename = secure_filename(image.filename)
                    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image.save(filename)
                    return render_template('page.html', uploaded_image=image.filename)
                except Exception as e:
                    return f'Erreur lors de l\'enregistrement de l\'image : {str(e)}'
            else:
                return 'Seules les images PNG sont autorisées.'
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)