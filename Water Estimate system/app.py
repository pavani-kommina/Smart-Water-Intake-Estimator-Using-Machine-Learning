from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import joblib
import pandas as pd
import numpy as np

# App config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Reethu2005'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# Extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect to login if not authenticated

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ML Model Loading (same as before)
model_path = 'models/water_intake_model.pkl'
scaler_path = 'models/scaler.pkl'
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("Run 'python ml_model.py' first to generate model files.")

# Routes
@app.route('/')
@login_required  # Protect the main page
def index():
    return render_template('index.html', user=current_user)  # This renders Jinja2!

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

# ADD THIS NEW ROUTE - This handles the form display
@app.route('/recommendation')
@login_required
def recommendation():
    return render_template('recommendation.html', user=current_user)

@app.route('/predict', methods=['POST'])
@login_required  # Protect prediction
def predict():
    try:
        print("Received request:", request.form)  # Debug
        data = request.form
        features = [
            float(data['age']),
            float(data['weight']),
            int(data['gender']),
            int(data['activity_level']),
            int(data['climate']),
            int(data['health']),
            int(data['diet_type']),
            float(data['sleep_hours'])
        ]
        
        print("Features extracted:", features)  # Debug
        
        input_df = pd.DataFrame([features], columns=[
            'age', 'weight', 'gender', 'activity_level', 'climate', 'health', 'diet_type', 'sleep_hours'
        ])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        print("Prediction:", prediction)  # Debug
        
        recommendation = f"{prediction:.2f} liters per day"
        
        return jsonify({
            'recommendation': recommendation,
            'details': f'Based on your inputs: Age {data["age"]}, Weight {data["weight"]}kg, etc. | User: {current_user.username}'
        })
    except KeyError as e:
        print(f"Missing form field: {e}")
        return jsonify({'error': f'Missing input: {str(e)}'}), 400
    except ValueError as e:
        print(f"Invalid input value: {e}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure instance folder exists
    os.makedirs('instance', exist_ok=True)
    
    # Create app context and DB with error handling
    try:
        with app.app_context():
            db.create_all()  # Create tables if they don't exist
            print("Database initialized successfully at instance/app.db")
    except Exception as e:
        print(f"Database creation error: {e}")
        print("Try creating 'instance' folder manually or check permissions.")
        # Optionally, exit or continue without DB (but auth won't work)
    
    app.run(debug=True)