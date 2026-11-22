import pandas as pd


def load_books(path="data/books.csv"):
    """Read the raw CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)

    # Keep only the columns we need, with friendlier names
    df = df[
        [
            "book_id",
            "title",
            "authors",
            "original_publication_year",
            "average_rating",
            "ratings_count",
            "language_code",
        ]
    ].rename(columns={"original_publication_year": "year"})

    # Drop rows missing key fields, then make year a whole number
    df = df.dropna(subset=["title", "authors", "year"])
    df["year"] = df["year"].astype(int)

    # Use only the first listed author as the "main" author
    df["main_author"] = df["authors"].str.split(",").str[0].str.strip()

    return df


def add_scores(df):
    """Add quality, visibility and hidden-gem scores to the books table."""
    df = df.copy()

    # Bayesian-adjusted rating: pulls books with few ratings toward the average
    C = df["average_rating"].mean()      # average rating of all books
    m = df["ratings_count"].median()     # "typical" number of ratings
    v = df["ratings_count"]
    R = df["average_rating"]
    df["bayes_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

    # Cohort = decade of publication (anything before 1900 is grouped together)
    df["cohort"] = (df["year"].clip(lower=1900) // 10) * 10

    # Percentile rank (0-100) of each book within its own cohort
    grouped = df.groupby("cohort")
    df["quality_pct"] = grouped["bayes_rating"].rank(pct=True) * 100
    df["visibility_pct"] = grouped["ratings_count"].rank(pct=True) * 100

    df["gem_score"] = df["quality_pct"] - df["visibility_pct"]
    return df