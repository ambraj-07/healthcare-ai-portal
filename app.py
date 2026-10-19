from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib

app = Flask(__name__)
app.secret_key = 'super_secret_healthcare_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    gender = db.Column(db.String(20), default="Not Specified")
    height = db.Column(db.Float, default=0.0) # in cm
    weight = db.Column(db.Float, default=0.0) # in kg
    blood_pressure = db.Column(db.String(20), default="120/80")
    heart_rate = db.Column(db.Integer, default=72)
    blood_sugar = db.Column(db.Integer, default=95)
    records = db.relationship('HealthRecord', backref='user', lazy=True)

class HealthRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(150), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    specialist = db.Column(db.String(150), nullable=False)
    urgency = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

def calculate_bmi(user):
    if user.height > 0 and user.weight > 0:
        return round(user.weight / ((user.height / 100.0) ** 2), 1)
    return 0

def get_clinical_guidance(disease_name):
    disease_lower = disease_name.lower()
    if any(term in disease_lower for term in ['heart', 'cardio', 'blood pressure', 'hypertension']):
        return {"specialist": "Cardiologist", "urgency": "High Risk", "badge_class": "badge-high"}
    elif any(term in disease_lower for term in ['skin', 'rash', 'acne', 'fungal', 'psoriasis']):
        return {"specialist": "Dermatologist", "urgency": "Low Risk", "badge_class": "badge-normal"}
    elif any(term in disease_lower for term in ['thyroid', 'diabetes', 'sugar']):
        return {"specialist": "Endocrinologist", "urgency": "Moderate Risk", "badge_class": "badge-moderate"}
    elif any(term in disease_lower for term in ['arthrit', 'joint', 'bone']):
        return {"specialist": "Rheumatologist / Orthopedic", "urgency": "Moderate Risk", "badge_class": "badge-moderate"}
    elif any(term in disease_lower for term in ['brain', 'neural', 'paralysis', 'migraine']):
        return {"specialist": "Neurologist", "urgency": "High Risk", "badge_class": "badge-high"}
    else:
        return {"specialist": "General Physician / Internal Medicine", "urgency": "Moderate Risk", "badge_class": "badge-moderate"}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid email or password.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', error="Email already registered.")
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    user_records = HealthRecord.query.filter_by(user_id=session['user_id']).all()
    bmi = calculate_bmi(user)
    return render_template('index.html', user=user, bmi=bmi, symptoms=model_columns, records=user_records, active_tab='checker')

@app.route('/update_vitals', methods=['POST'])
def update_vitals():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    user.gender = request.form.get('gender', user.gender)
    user.height = float(request.form.get('height') or user.height)
    user.weight = float(request.form.get('weight') or user.weight)
    user.blood_pressure = request.form.get('blood_pressure', user.blood_pressure)
    user.heart_rate = int(request.form.get('heart_rate') or user.heart_rate)
    user.blood_sugar = int(request.form.get('blood_sugar') or user.blood_sugar)
    db.session.commit()
    return redirect(url_for('home'))

@app.route('/clear_history')
def clear_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    HealthRecord.query.filter_by(user_id=session['user_id']).delete()
    db.session.commit()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    user_records = HealthRecord.query.filter_by(user_id=session['user_id']).all()
    bmi = calculate_bmi(user)

    try:
        selected_symptoms = request.form.getlist('symptoms')
        if len(selected_symptoms) < 3:
            return render_template('index.html', user=user, bmi=bmi, symptoms=model_columns, records=user_records, error_text="Please select at least 3 symptoms.", active_tab='checker')
        
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in model_columns]
        probabilities = model.predict_proba([input_vector])[0]
        classes = model.classes_
        top_preds = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:3]
        
        results = []
        for pred in top_preds:
            disease = pred[0]
            prob = round(pred[1] * 100, 2)
            guidance = get_clinical_guidance(disease)
            results.append({
                "disease": disease, "probability": prob,
                "specialist": guidance["specialist"], "urgency": guidance["urgency"], "badge_class": guidance["badge_class"]
            })
        
        top_match = results[0]
        new_record = HealthRecord(
            disease=top_match['disease'], probability=top_match['probability'],
            specialist=top_match['specialist'], urgency=top_match['urgency'], user_id=user.id
        )
        db.session.add(new_record)
        db.session.commit()
        
        user_records = HealthRecord.query.filter_by(user_id=user.id).all()
        return render_template('index.html', user=user, bmi=bmi, symptoms=model_columns, predictions=results, records=user_records, active_tab='checker')
    
    except Exception as e:
        return render_template('index.html', user=user, bmi=bmi, symptoms=model_columns, records=user_records, error_text=f"Error: {str(e)}", active_tab='checker')

@app.route('/doctors')
def doctors():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    bmi = calculate_bmi(user)
    
    # Expanded Directory of 8 Verified Specialists
    doctor_list = [
        {
            "name": "Dr. R. K. Sharma", 
            "specialty": "Cardiologist", 
            "qualification": "MD, DM (Cardiology)", 
            "availability": "Mon - Fri (10 AM - 2 PM)",
            "phone": "+91 651 249 1122",
            "location": "Medanta Specialist Wing, Ranchi"
        },
        {
            "name": "Dr. Anjali Sen", 
            "specialty": "Endocrinologist", 
            "qualification": "MD, DNB (Endocrinology)", 
            "availability": "Tue - Sat (11 AM - 4 PM)",
            "phone": "+91 651 233 5544",
            "location": "HealthCore Clinic, Main Road, Ranchi"
        },
        {
            "name": "Dr. Manish Kumar", 
            "specialty": "Neurologist", 
            "qualification": "MBBS, DM (Neurology)", 
            "availability": "Mon, Wed, Fri (9 AM - 1 PM)",
            "phone": "+91 651 254 8899",
            "location": "RIMS Campus Rd, Bariatu, Ranchi"
        },
        {
            "name": "Dr. Priya Verma", 
            "specialty": "Dermatologist", 
            "qualification": "MD (Skin & VD)", 
            "availability": "Mon - Sat (10 AM - 6 PM)",
            "phone": "+91 651 220 3311",
            "location": "Skin & Laser Care Centre, Circular Road"
        },
        {
            "name": "Dr. Suresh Mehta", 
            "specialty": "General Physician / Internal Medicine", 
            "qualification": "MBBS, MD (Medicine)", 
            "availability": "Mon - Sat (9 AM - 8 PM)",
            "phone": "+91 651 231 7700",
            "location": "City General Hospital, Morabadi, Ranchi"
        },
        {
            "name": "Dr. Neha Sinha", 
            "specialty": "Rheumatologist / Orthopedic", 
            "qualification": "MS (Orthopedics), MRCS", 
            "availability": "Mon - Sat (10 AM - 5 PM)",
            "phone": "+91 651 248 3322",
            "location": "OrthoCare Clinic, Kanke Road, Ranchi"
        },
        {
            "name": "Dr. Alok Mukherjee", 
            "specialty": "Pulmonologist", 
            "qualification": "MD (Respiratory Medicine)", 
            "availability": "Tue, Thu, Sat (2 PM - 7 PM)",
            "phone": "+91 651 236 9911",
            "location": "Chest & Allergy Clinic, Lalpur, Ranchi"
        },
        {
            "name": "Dr. Swati Roy", 
            "specialty": "Pediatrician", 
            "qualification": "MD (Pediatrics), DCH", 
            "availability": "Mon - Sat (9 AM - 2 PM)",
            "phone": "+91 651 250 4488",
            "location": "Child Care Center, Harmu Housing Colony"
        }
    ]
    return render_template('index.html', user=user, bmi=bmi, doctors=doctor_list, active_tab='doctors')

@app.route('/hospitals')
def hospitals():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    bmi = calculate_bmi(user)
    
    # Expanded Directory of 6 Accredited Partner Hospitals
    hospital_list = [
        {
            "name": "Medanta Abdur Razzaq Ansari Memorial Hospital", 
            "location": "Irba, Ranchi, Jharkhand", 
            "contact": "+91 651 712 3000", 
            "emergency": "24/7 Available"
        },
        {
            "name": "RIMS (Rajendra Institute of Medical Sciences)", 
            "location": "Bariatu, Ranchi, Jharkhand", 
            "contact": "+91 651 254 1533", 
            "emergency": "24/7 Trauma Center"
        },
        {
            "name": "Boring Road Nursing Home & Diagnostic", 
            "location": "Main Road Central Hub, Ranchi", 
            "contact": "+91 651 233 4455", 
            "emergency": "Available"
        },
        {
            "name": "Santevita Hospital", 
            "location": "Hinoo, Ranchi, Jharkhand", 
            "contact": "+91 651 225 2200", 
            "emergency": "24/7 Multi-Specialty"
        },
        {
            "name": "Orchid Medical Centre", 
            "location": "Hazaribagh Road, Ranchi", 
            "contact": "+91 651 221 1500", 
            "emergency": "24/7 Critical Care"
        },
        {
            "name": "Batra Hospital & Research Centre (Regional Wing)", 
            "location": "Kanke Road, Ranchi", 
            "contact": "+91 651 277 8800", 
            "emergency": "24/7 Emergency & ICU"
        }
    ]
    return render_template('index.html', user=user, bmi=bmi, hospitals=hospital_list, active_tab='hospitals')

if __name__ == '__main__':
    app.run(debug=True)