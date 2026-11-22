import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_prep import load_books, add_scores
from src.model import ALL_FEATURES, add_model_features, explain_row, train_and_evaluate

st.set_page_config(page_title="BookScope Intelligence", layout="wide")


@st.cache_data
def get_books():
    return add_model_features(add_scores(load_books()))


books = get_books()


@st.cache_resource
def get_model_results():
    # cache_resource keeps trained model objects in memory, so we train only once
    return train_and_evaluate(books)


page = st.sidebar.radio("Page", ["Overview", "Hidden Gems", "Bestseller Predictor"])
st.title("BookScope Intelligence")

if page == "Overview":
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Books", f"{len(books):,}")
    col2.metric("Authors", f"{books['main_author'].nunique():,}")
    col3.metric("Avg rating", f"{books['average_rating'].mean():.2f}")

    st.dataframe(books.head(50), width="stretch")

elif page == "Hidden Gems":
    st.subheader("Hidden Gems")
    st.write(
        "Books readers rate highly, but that few readers have found, "
        "compared with other books from the same decade."
    )

    min_q = st.slider("Minimum quality percentile", 50, 99, 80)
    max_v = st.slider("Maximum visibility percentile", 1, 90, 50)

    df = books.copy()
    df["is_gem"] = (df["quality_pct"] >= min_q) & (df["visibility_pct"] <= max_v)
    df["label"] = df["is_gem"].map({True: "Hidden gem", False: "Other"})
    gems = df[df["is_gem"]].sort_values("gem_score", ascending=False)

    st.metric("Hidden gems found", f"{len(gems):,}")

    fig = px.scatter(
        df,
        x="visibility_pct",
        y="quality_pct",
        color="label",
        hover_data=["title", "main_author", "year"],
        opacity=0.6,
        labels={"visibility_pct": "Visibility percentile", "quality_pct": "Quality percentile"},
    )
    st.plotly_chart(fig, width="stretch")

    st.write("Top gems by gem score")
    cols = ["title", "main_author", "year", "average_rating", "ratings_count",
            "quality_pct", "visibility_pct", "gem_score"]
    st.dataframe(gems[cols].head(50).round(1), width="stretch")

else:
    st.subheader("Bestseller Predictor (pre-publication)")
    st.write(
        "Estimates the chance a new book becomes a bestseller using only information "
        "known before publication. 'Bestseller' here is a proxy: top 20% by popularity "
        "within the book's decade."
    )

    with st.spinner("Training models (first load only)..."):
        res = get_model_results()

    c1, c2, c3 = st.columns(3)
    c1.metric("Training books (1990-2010)", f"{res['n_train']:,}")
    c2.metric("Test books (2011 onward)", f"{res['n_test']:,}")
    c3.metric("Bestseller rate in test set", f"{res['base_rate']:.1%}")

    st.markdown("#### Model comparison on the later period")
    st.dataframe(res["models"].round(3), width="stretch", hide_index=True)
    st.caption(
        f"Random guessing would score AUC 0.50 and average precision {res['base_rate']:.3f}."
    )

    st.markdown("#### Does the author's earlier record matter?")
    fig_ab = px.bar(res["ablation"].round(3), x="Feature set", y="AUC", range_y=[0.4, 1.0], text="AUC")
    st.plotly_chart(fig_ab, width="stretch")

    st.markdown("#### What-if simulator")
    left, right = st.columns(2)
    with left:
        year = st.slider("Planned publication year", 1990, 2030, 2015)
        is_english = st.checkbox("English-language book", value=True)
        prev_books = st.number_input("Earlier books by this author", 0, 50, 0)
        if prev_books > 0:
            prev_avg = st.slider("Average rating of earlier books", 3.0, 4.8, 4.0, 0.05)
            prev_best = st.slider("Share of earlier books that were bestsellers", 0.0, 1.0, 0.2, 0.05)
            typical_ratings = st.number_input(
                "Typical number of ratings on earlier books", 0, 5_000_000, 20_000, step=1_000
            )
            prev_log = float(np.log1p(typical_ratings))
        else:
            st.info("Debut author: no earlier record available.")
            prev_avg = float(books["average_rating"].mean())
            prev_best, prev_log = 0.0, 0.0

    row = pd.DataFrame([{
        "year": year,
        "is_english": int(is_english),
        "prev_books": prev_books,
        "prev_avg_rating": prev_avg,
        "prev_best_rate": prev_best,
        "prev_log_ratings": prev_log,
    }])[ALL_FEATURES]

    prob = res["xgb"].predict_proba(row)[0, 1]
    contrib = explain_row(res["xgb"], row)

    with right:
        st.metric("Predicted bestseller probability", f"{prob:.1%}")
        cdf = contrib.sort_values().reset_index()
        cdf.columns = ["Feature", "Effect"]
        cdf["Direction"] = np.where(cdf["Effect"] > 0, "Raises chance", "Lowers chance")
        fig_c = px.bar(
            cdf, x="Effect", y="Feature", orientation="h", color="Direction",
            color_discrete_map={"Raises chance": "#2ca02c", "Lowers chance": "#d62728"},
        )
        st.plotly_chart(fig_c, width="stretch")
        st.caption("Bars show how much each input pushed this prediction up or down (SHAP values, log-odds).")