import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# HTML Template with Embedded Modern CSS, Cards, Floating Shadows & Glowing Effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Student Score Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-gradient: radial-gradient(circle at top right, #1e1b4b, #0f172a 50%, #020617);
      --card-bg: rgba(30, 41, 59, 0.72);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.45);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --input-bg: rgba(15, 23, 42, 0.65);
      --input-border: rgba(255, 255, 255, 0.12);
      --radius: 16px;
      --shadow-soft: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 25px -5px rgba(99, 102, 241, 0.2);
      --shadow-input: inset 0 2px 4px rgba(0, 0, 0, 0.4);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', sans-serif;
    }

    body {
      min-height: 100vh;
      background: var(--bg-gradient);
      color: var(--text-main);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 40px 20px;
    }

    .container {
      width: 100%;
      max-width: 900px;
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      box-shadow: var(--shadow-soft);
      padding: 40px;
      position: relative;
      overflow: hidden;
    }

    .container::before {
      content: '';
      position: absolute;
      top: -120px;
      right: -120px;
      width: 250px;
      height: 250px;
      background: var(--accent-glow);
      filter: blur(80px);
      pointer-events: none;
      border-radius: 50%;
    }

    header {
      margin-bottom: 30px;
      text-align: center;
    }

    h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    p.subtitle {
      color: var(--text-muted);
      font-size: 0.95rem;
      margin-top: 6px;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }

    .input-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: #cbd5e1;
      text-transform: capitalize;
    }

    input, select {
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 0.95rem;
      color: var(--text-main);
      box-shadow: var(--shadow-input);
      transition: all 0.2s ease;
      outline: none;
    }

    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    select option {
      background-color: #0f172a;
      color: #f8fafc;
    }

    .btn-submit {
      width: 100%;
      background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
      color: white;
      border: none;
      padding: 14px 28px;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 10px;
      cursor: pointer;
      box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
      transition: all 0.3s ease;
      position: relative;
    }

    .btn-submit:hover {
      transform: translateY(-2px);
      box-shadow: 0 14px 28px -5px rgba(99, 102, 241, 0.65);
      filter: brightness(1.08);
    }

    .btn-submit:active {
      transform: translateY(0);
    }

    .result-badge {
      margin-top: 30px;
      padding: 24px;
      border-radius: 12px;
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.3);
      text-align: center;
      box-shadow: inset 0 0 20px rgba(99, 102, 241, 0.1);
      animation: fadeIn 0.4s ease;
    }

    .result-badge span {
      display: block;
      color: var(--text-muted);
      font-size: 0.9rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .result-badge strong {
      font-size: 2.5rem;
      font-weight: 700;
      color: #818cf8;
      display: inline-block;
      margin-top: 4px;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>

  <div class="container">
    <header>
      <h1>Performance Predictor</h1>
      <p class="subtitle">SVR-Powered Academic Outcome Assessment Engine</p>
    </header>

    <form method="POST" action="/">
      <div class="form-grid">
        <div class="input-group">
          <label for="student_id">Student ID</label>
          <input type="number" step="any" id="student_id" name="student_id" value="{{ form_data.get('student_id', '101') }}" required />
        </div>

        <div class="input-group">
          <label for="age">Age</label>
          <input type="number" step="any" id="age" name="age" value="{{ form_data.get('age', '20') }}" required />
        </div>

        <div class="input-group">
          <label for="gender">Gender</label>
          <select id="gender" name="gender">
            <option value="0" {% if form_data.get('gender') == '0' %}selected{% endif %}>Female (0)</option>
            <option value="1" {% if form_data.get('gender', '1') == '1' %}selected{% endif %}>Male (1)</option>
          </select>
        </div>

        <div class="input-group">
          <label for="course">Course Code</label>
          <input type="number" step="any" id="course" name="course" value="{{ form_data.get('course', '1') }}" required />
        </div>

        <div class="input-group">
          <label for="study_hours">Daily Study Hours</label>
          <input type="number" step="any" id="study_hours" name="study_hours" value="{{ form_data.get('study_hours', '4.5') }}" required />
        </div>

        <div class="input-group">
          <label for="class_attendance">Class Attendance (%)</label>
          <input type="number" step="any" id="class_attendance" name="class_attendance" value="{{ form_data.get('class_attendance', '85') }}" required />
        </div>

        <div class="input-group">
          <label for="internet_access">Internet Access</label>
          <select id="internet_access" name="internet_access">
            <option value="1" {% if form_data.get('internet_access', '1') == '1' %}selected{% endif %}>Yes (1)</option>
            <option value="0" {% if form_data.get('internet_access') == '0' %}selected{% endif %}>No (0)</option>
          </select>
        </div>

        <div class="input-group">
          <label for="sleep_hours">Sleep Hours</label>
          <input type="number" step="any" id="sleep_hours" name="sleep_hours" value="{{ form_data.get('sleep_hours', '7') }}" required />
        </div>

        <div class="input-group">
          <label for="sleep_quality">Sleep Quality (1 - 5)</label>
          <input type="number" step="any" id="sleep_quality" name="sleep_quality" value="{{ form_data.get('sleep_quality', '4') }}" required />
        </div>

        <div class="input-group">
          <label for="study_method">Study Method</label>
          <input type="number" step="any" id="study_method" name="study_method" value="{{ form_data.get('study_method', '2') }}" required />
        </div>

        <div class="input-group">
          <label for="facility_rating">Facility Rating (1 - 5)</label>
          <input type="number" step="any" id="facility_rating" name="facility_rating" value="{{ form_data.get('facility_rating', '4') }}" required />
        </div>

        <div class="input-group">
          <label for="exam_difficulty">Exam Difficulty (1 - 5)</label>
          <input type="number" step="any" id="exam_difficulty" name="exam_difficulty" value="{{ form_data.get('exam_difficulty', '3') }}" required />
        </div>
      </div>

      <button type="submit" class="btn-submit">Run Prediction</button>
    </form>

    {% if prediction is not none %}
    <div class="result-badge">
      <span>Estimated Prediction Score</span>
      <strong>{{ "%.2f"|format(prediction) }}</strong>
    </div>
    {% endif %}
  </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}
    if request.method == "POST":
        form_data = request.form
        features = [
            float(request.form.get("student_id", 0)),
            float(request.form.get("age", 0)),
            float(request.form.get("gender", 0)),
            float(request.form.get("course", 0)),
            float(request.form.get("study_hours", 0)),
            float(request.form.get("class_attendance", 0)),
            float(request.form.get("internet_access", 0)),
            float(request.form.get("sleep_hours", 0)),
            float(request.form.get("sleep_quality", 0)),
            float(request.form.get("study_method", 0)),
            float(request.form.get("facility_rating", 0)),
            float(request.form.get("exam_difficulty", 0)),
        ]
        
        arr = np.array([features])
        pred_value = model.predict(arr)[0]
        prediction = float(pred_value)

    return render_template_string(HTML_TEMPLATE, prediction=prediction, form_data=form_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
