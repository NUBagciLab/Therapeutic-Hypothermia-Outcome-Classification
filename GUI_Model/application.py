from flask import Flask, request, render_template
import pandas as pd
import pickle
import json
import numpy as np

with open('tuned_TH_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__) 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        GA = float(request.form['GA'])
        creatinine = float(request.form['creatinine'])
        creatinine_unit = request.form['creatinine_unit']
        PNA = float(request.form['PNA'])
        BW = float(request.form['BW'])
    except ValueError as e:
        chart_data = json.dumps({"labels": [], "data": []})
        return render_template('index.html', error_message=str(e), chart_data=chart_data)

    if creatinine_unit == 'µmol/L':
        creatinine = creatinine / 88.4

    if not (34 <= GA <= 43 and 0 <= creatinine <= 10 and 0 <= PNA <= 28 and 1500 <= BW <= 5000):
        error_message = 'Invalid input. Please try again with values in a more reasonable range.'
        chart_data = json.dumps({"labels": [], "data": []})
        return render_template('index.html', error_message=error_message, chart_data=chart_data)

    GA_BW_interaction = GA * BW
    GA_Creatinine_interaction = GA * creatinine
    PNA_Creatinine_interaction = PNA * creatinine

    input_df = pd.DataFrame({
        'GA (weeks)': [GA],
        'BW (grams)': [BW],
        'PNA (days)': [PNA],
        'creatinine (mg/dL)': [creatinine],
        'GA_BW_interaction': [GA_BW_interaction],
        'GA_Creatinine_interaction': [GA_Creatinine_interaction],
        'PNA_Creatinine_interaction': [PNA_Creatinine_interaction]
    })

    final_prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)
    confidence_percent = round(max(prediction_proba[0]) * 100, 2)

    confidence_levels = [round(prob * 100, 2) for prob in prediction_proba[0]]

    labels = ["Patient survives TH", "Patient survives TH with AKI", "Patient fatality from TH", "Patient fatality, suffers AKI from TH", "Patient does not require TH"]
    chart_data = json.dumps({"labels": labels, "data": confidence_levels})

    user_input = {
        'GA': GA,
        'creatinine': creatinine,
        'creatinine_unit': creatinine_unit,
        'PNA': PNA,
        'BW': BW
    }

    prediction_map = {
        0: "This infant may require therapeutic hypothermia treatment and is likely to survive without issue.",
        1: "If therapeutic hypothermia is induced upon this infant, survival of the infant is likely, although there is risk of acute kidney injury developing.",
        2: "Fatality may result during therapeutic hypothermia treatment of this infant.",
        3: "There is a risk of acute kidney injury as well as death while this infant undergoes therapeutic hypothermia treatment.",
        4: "It is likely that this infant does not require therapeutic hypothermia treatment."
    }
    prediction_text = f"{prediction_map.get(final_prediction)}"

    return render_template('index.html', prediction_text=prediction_text,
                           confidence=confidence_percent, chart_data=chart_data,
                           user_input=user_input)


if __name__ == "__main__":
    app.run(debug=True)
