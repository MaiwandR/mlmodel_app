import numpy as np
from flask import Flask, render_template, request, send_from_directory
import pickle as pk


app = Flask(__name__)

# Load the trained model
model = pk.load(open('model.joblib', 'rb'))

def calculators(input):

    finalized_data = [0]*30

    features_ranges = {
        0 : {'min': 6.98, 'max':28.1},
        1 : {'min': 9.71, 'max':39.3},
        2 : {'min': 43.8, 'max':189},
        3 : {'min': 144, 'max': 189},
        4 : {'min': 144, 'max':2501},
        5 : {'min': 0.05, 'max':0.16},
        6 : {'min': 6.98, 'max':28.1},
        7 : {'min': 0, 'max':0.35},
        8 : {'min': 0, 'max':0.2},
        9 : {'min': 0.11, 'max':0.3},
        10 : {'min': 0.05, 'max':0.1},
        11 : {'min': 0.11, 'max':2.87},
        12 : {'min': 0.36, 'max':4.88},
        13 : {'min': 0.76, 'max':22},
        14 : {'min': 6.8, 'max':542},
        15 : {'min': 0, 'max':0.03},
        16 : {'min': 0, 'max':0.14},
        17 : {'min': 0, 'max':0.4},
        18 : {'min': 0, 'max':0.05},
        19 : {'min': 0.01, 'max':0.08},
        20 : {'min': 0, 'max':0.03},
        21 : {'min': 7.93, 'max':36},
        22 : {'min': 12, 'max':49.5},
        23 : {'min': 50.4, 'max':251},
        24 : {'min': 185, 'max':4255},
        25 : {'min': 0.07, 'max':1.06},
        26 : {'min': 0, 'max':1.26},
        27 : {'min': 0, 'max':.29},
        28 : {'min': 0.16, 'max':.66},
        29 : {'min': 0.06, 'max':.21}, 
    }

    questions_indices = {
        1 : [0,2,3,20,22,23],
        2 : [1, 4 , 21,24 ],
        3 : [5,25],
        4 : [8,28],
        5 : [10,12,13],
        6 : [11,14],
        7 : [15,16,17],
        8 : [18,19],
        9 : [6,7,26,27],
        10 : [9,29],
    }

    # Scale and map the inputs to the right indices
    for i, val in enumerate(input):
        for index in questions_indices[i+1]:
            min_val = features_ranges[index]['min']  
            max_val = features_ranges[index]['max']

            # Apply Min-Max Scaling
            x = round(min_val + ((val - 1) / 9) * (max_val - min_val), 5)

            # Place scaled value at the correct index
            finalized_data[index] = x

    
    return finalized_data



# Home route to render the input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract 30 features from form data
        features = [float(request.form[f'feature{i}']) for i in range(1,11)]
        
        fixed_features = calculators(features)
        # Convert to NumPy array for model input
        input_data = np.array([fixed_features])
        
        # Make prediction (M or B)
        prediction = model.predict(input_data)[0]
        
        # Return the result to the frontend
        return render_template('index.html', prediction = prediction)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
