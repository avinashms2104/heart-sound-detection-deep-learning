# Heart Sound Detection Using Machine Learning

This project demonstrates a simple machine learning approach to classify heart sounds as **normal or abnormal** using simulated phonocardiogram features.

## Project Overview
Heart sound analysis can assist in early detection of cardiovascular abnormalities. In this project, a dataset containing signal characteristics is used to train a classification model.

## Dataset
The dataset contains the following features:

- signal_strength
- frequency
- noise_level
- label (0 = normal, 1 = abnormal)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Model
A **Random Forest Classifier** is trained on the dataset to classify heart sounds.

## How to Run
1. Install requirements:
```
pip install -r requirements.txt
```

2. Run the model:
```
python heart_sound_model.py
```

## Project Structure
```
heart-sound-detection-deep-learning/
│
├── heart_sound_dataset.csv
├── heart_sound_model.py
├── requirements.txt
└── README.md
```

## Author
Avinash
Master of Data Science Student
University of Auckland