import streamlit as st
import pandas as pd
import pickle


# Load sentiment model and TF-IDF
sentiment_model = pickle.load(open("sentiment.pkl", "rb"))
sentiment_tfidf = pickle.load(open("sentiment_tfidf.pkl", "rb"))

# Load fake review model and TF-IDF
fake_model = pickle.load(open("fake_review_model.pkl", "rb"))
fake_tfidf = pickle.load(open("fake_review_tfidf.pkl", "rb"))

st.title("Review Authenticity and Sentiment Analysis")
st.write("Upload a CSV file containing reviews to analyze authenticity and sentiment.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "review_text" not in df.columns:
        st.error("CSV must contain a column named 'review_text'")
    else:
        st.success("File uploaded successfully")

        X_fake = fake_tfidf.transform(df["review_text"])
        fake_probs = fake_model.predict_proba(X_fake)[:, 1]

        threshold = 0.35
        df["is_fake"] = (fake_probs >= threshold).astype(int)

        genuine_df = df[df["is_fake"] == 0].copy()

        X_sent = sentiment_tfidf.transform(genuine_df["review_text"])
        genuine_df["sentiment"] = sentiment_model.predict(X_sent)

        genuine_df["sentiment_label"] = genuine_df["sentiment"].map({
            1: "Positive",
            0: "Negative"
        })

        st.subheader("Summary")

        total_reviews = len(df)
        fake_count = df["is_fake"].sum()
        genuine_count = total_reviews - fake_count

        pos_count = (genuine_df["sentiment"] == 1).sum()
        neg_count = (genuine_df["sentiment"] == 0).sum()

        st.write(f"Total Reviews: {total_reviews}")
        st.write(f"Genuine Reviews: {genuine_count}")
        st.write(f"Potentially Fake Reviews: {fake_count}")
        st.write(f"Positive Reviews (Genuine): {pos_count}")
        st.write(f"Negative Reviews (Genuine): {neg_count}")

        st.subheader("Sentiment Distribution (Genuine Reviews)")

        sentiment_chart_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [pos_count, neg_count]
        })

        st.bar_chart(sentiment_chart_df.set_index("Sentiment"))


        st.subheader("Review Analysis")

        st.subheader("Genuine Reviews")

        for _, row in genuine_df.iterrows():
            if row["sentiment_label"] == "Positive":
                st.markdown(
    f"""
    <div style="
        padding:12px;
        border-radius:8px;
        background-color:#e8f5e9;
        margin-bottom:12px;
        color:#1b5e20;
    ">
        <strong style="color:#2e7d32;">Positive Review</strong><br>
        <span style="color:#1b5e20;">{row['review_text']}</span>
    </div>
    """,
    unsafe_allow_html=True
)

            else:
             st.markdown(
    f"""
    <div style="
        padding:12px;
        border-radius:8px;
        background-color:#fdecea;
        margin-bottom:12px;
        color:#7f1d1d;
    ">
        <strong style="color:#c62828;">Negative Review</strong><br>
        <span style="color:#7f1d1d;">{row['review_text']}</span>
    </div>
    """,
    unsafe_allow_html=True
)

