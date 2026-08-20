from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the entire preprocessing and modeling pipeline
pipeline = joblib.load('student_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Grab all inputs from the HTML form
        form_data = {
            'Gender': request.form['Gender'],
            'EthnicGroup': request.form['EthnicGroup'],
            'ParentEduc': request.form['ParentEduc'],
            'LunchType': request.form['LunchType'],
            'TestPrep': request.form['TestPrep'],
            'ParentMaritalStatus': request.form['ParentMaritalStatus'],
            'PracticeSport': request.form['PracticeSport'],
            'IsFirstChild': request.form['IsFirstChild'],
            'NrSiblings': int(request.form['NrSiblings']),
            'TransportMeans': request.form['TransportMeans'],
            'WklyStudyHours': request.form['WklyStudyHours'],
            'ReadingScore': float(request.form['ReadingScore']),
            'WritingScore': float(request.form['WritingScore'])
        }
        
        # 2. Convert dictionary to a DataFrame (Pipeline requirement)
        input_df = pd.DataFrame([form_data])
        
        # 3. Predict the Math Score
        prediction = pipeline.predict(input_df)[0]
        
        # 4. Generate Insights based on user input
        insights = []
        if request.form['TestPrep'] == 'none':
            insights.append("🚀 Consider completing a Test Preparation course; this is one of the highest predictors of score improvement.")
        if request.form['WklyStudyHours'] == '< 5':
            insights.append("📚 Increasing your weekly study time to 5-10 hours could significantly boost your performance.")
        if float(request.form['ReadingScore']) < 60:
            insights.append("📖 Your reading score is a bit low. Improving reading speed and comprehension often benefits math word-problem performance.")
        
        # 5. Return prediction AND insights to the template
        return render_template(
            'index.html', 
            prediction_text=f'Predicted Math Score: {prediction:.1f} / 100',
            prediction_score=prediction,  # ADD THIS LINE for the gauge chart!
            insights=insights
        )

if __name__ == '__main__':
    app.run(debug=True)