Movie Genre Classification
This project predicts the genre of a movie based on its plot summary using Machine Learning.

Project Overview
- Input: Movie plot description
- Output: Predicted genre (Action, Comedy, Drama, etc.)

Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization

Model Used
- Naive Bayes Classifier

Accuracy
- Achieved accuracy: ~47%

Features
- Text preprocessing (cleaning)
- TF-IDF feature extraction
- Genre prediction
- Custom input prediction

How to Run
1. Install dependencies:

pip install -r requirements.txt

2. Run the project:

python main.py

Project Structure
- data/ → dataset
- main.py → main code
- model.pkl → trained model
- tfidf.pkl → vectorizer

Future Improvements
- Improve accuracy using advanced models
- Build a web interface
- Use deep learning (LSTM, BERT)