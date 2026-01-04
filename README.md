# review-authenticity-sentiment-analysis
An end to end Machine Learning project that detects potentially fake reviews and performs sentiment analysis on genuine reviews using NLP techniques.

## Datasets Used

Due to large file size limitations, the datasets used for this project are not included in the repository.

The following publicly available datasets were used during model development:

### Product Reviews Dataset
- Amazon Product Reviews  
- Source: https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews
- Description: Contains large scale product review text data used for training and evaluating the sentiment analysis model.

### Fake Reviews Dataset
- Fake Company Reviews  
- Source: https://www.kaggle.com/datasets/sachinsk/fake-company-reviews
- Description: Contains company review text with proxy labels used to train the fake review detection model.

You can download these datasets directly from Kaggle or test the application using their own CSV files containing review text.


---

## Machine Learning and NLP Approach

### Text Vectorization
- TF IDF (Term Frequency Inverse Document Frequency) is used to convert raw review text into numerical feature vectors.

### Fake Review Detection Model
- A linear classification model (Logistic Regression) is trained using proxy labels.
- The model outputs probabilities, and a decision threshold is applied to conservatively flag potentially fake reviews.

### Sentiment Analysis Model
- A linear text classification model is trained to classify reviews as Positive or Negative.
- Sentiment analysis is performed only on reviews classified as genuine.

This design helps reduce noise in sentiment analysis caused by suspicious or low quality reviews.

---

## Tech Stack

- Python
- Scikit learn
- NLP with TF IDF
- Streamlit
- Pandas and NumPy

---

## Purpose of the Project

This project is not intended for casual end users.  
It is designed for:
- Machine Learning learners
- NLP practitioners
- Students and researchers
- Anyone interested in understanding practical ML pipelines

---

## How to Run the Project

1. Install dependencies 
2. Run the Streamlit app  
3. Upload a CSV file containing review text to analyze authenticity and sentiment.

---

## Key Learnings

- Handling noisy and proxy labeled data
- Designing conservative ML systems
- Working with text based features
- Building complete ML pipelines from model training to deployment

---

## Future Improvements

- Better fake review labeling strategies
- More advanced NLP models
- Improved visualization and analytics



