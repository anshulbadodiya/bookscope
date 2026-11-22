import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.auth import (
    SELF_SERVE_ROLES,
    list_unverified_readers,
    login_user,
    register_user,
    seed_admin,
    verify_user,
)
from src.data_prep import add_scores, load_books
from src.db import init_db
from src.demo import DEMO_PASSWORD, DEMO_USERS, load_demo_data
from src.marketplace import (
    ASPECTS,
    add_manuscript,
    author_selection,
    cancel_and_refund,
    commission_beta_read,
    is_verified,
    list_available_authors,
    list_escrow_ledger,
    list_manuscripts,
    list_my_selections,
    list_requests_for_author,
    list_requests_for_reader,
    list_verified_readers,
    release_payment,
    select_author,
    submit_feedback,
)
from src.model import ALL_FEATURES, add_model_features, explain_row, train_and_evaluate

st.set_page_config(page_title="BookScope Intelligence", layout="wide")


@st.cache_resource
def setup_database():
    init_db()
    seed_admin()
    return True


setup_database()


@st.cache_data
def get_books():
    return add_model_features(add_scores(load_books()))


books = get_books()


@st.cache_resource
def get_model_results():
    return train_and_evaluate(books)


def flash(ok, message):
    """Remember a message across the st.rerun() that follows a button click."""
    st.session_state["flash"] = (ok, message)


def show_flash():
    if "flash" in st.session_state:
        ok, message = st.session_state.pop("flash")
        (st.success if ok else st.error)(message)


# ---------------------------------------------------------------- analytics pages
def page_overview():
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Books", f"{len(books):,}")
    col2.metric("Authors", f"{books['main_author'].nunique():,}")
    col3.metric("Avg rating", f"{books['average_rating'].mean():.2f}")
    st.dataframe(books.head(50), width="stretch")


def page_hidden_gems():
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
        df, x="visibility_pct", y="quality_pct", color="label",
        hover_data=["title", "main_author", "year"], opacity=0.6,
        labels={"visibility_pct": "Visibility percentile", "quality_pct": "Quality percentile"},
    )
    st.plotly_chart(fig, width="stretch")

    st.write("Top gems by gem score")
    cols = ["title", "main_author", "year", "average_rating", "ratings_count",
            "quality_pct", "visibility_pct", "gem_score"]
    st.dataframe(gems[cols].head(50).round(1), width="stretch")


def page_predictor():
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
    st.caption(f"Random guessing would score AUC 0.50 and average precision {res['base_rate']:.3f}.")

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
        "year": year, "is_english": int(is_english), "prev_books": prev_books,
        "prev_avg_rating": prev_avg, "prev_best_rate": prev_best, "prev_log_ratings": prev_log,
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


# ---------------------------------------------------------------- account pages
def page_login():
    st.subheader("Log in")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        user = login_user(email, password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Wrong email or password.")


def page_register():
    st.subheader("Create an account")
    with st.form("register_form"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password (min 8 characters)", type="password")
        role = st.selectbox("I am a...", SELF_SERVE_ROLES)
        submitted = st.form_submit_button("Register")
    if submitted:
        ok, message = register_user(name, email, password, role)
        (st.success if ok else st.error)(message)


def page_admin():
    st.subheader("Admin: verify readers")
    show_flash()
    pending = list_unverified_readers()
    if not pending:
        st.success("No readers are waiting for verification.")
    for r in pending:
        a, b = st.columns([4, 1])
        a.write(f"**{r['name']}**  ({r['email']})  registered {r['created_at']}")
        if b.button("Verify", key=f"verify_{r['id']}"):
            verify_user(r["id"])
            st.rerun()

    st.markdown("#### Escrow ledger (test mode)")
    ledger = list_escrow_ledger()
    if ledger:
        st.dataframe(pd.DataFrame(ledger), width="stretch", hide_index=True)
    else:
        st.caption("No escrow entries yet.")

    st.markdown("#### Demo data")
    st.caption(
        "Creates sample accounts, two manuscripts and one completed beta read, "
        "so the marketplace can be demonstrated quickly."
    )
    if st.button("Load demo data"):
        ok, msg = load_demo_data()
        flash(ok, msg)
        st.rerun()
    demo = pd.DataFrame(DEMO_USERS, columns=["Name", "Email", "Role"])
    demo["Password"] = DEMO_PASSWORD
    st.dataframe(demo, width="stretch", hide_index=True)


# ---------------------------------------------------------------- marketplace pages
def page_my_manuscripts():
    user = st.session_state["user"]
    st.subheader("My manuscripts")
    show_flash()

    publisher = author_selection(user["id"])
    if publisher:
        st.success(f"You have been selected by publisher **{publisher}**.")

    with st.form("manuscript_form", clear_on_submit=True):
        st.markdown("**Add a manuscript**")
        title = st.text_input("Title")
        description = st.text_area("Short description (what is it about?)")
        if st.form_submit_button("Save manuscript"):
            ok, msg = add_manuscript(user["id"], title, description)
            flash(ok, msg)
            st.rerun()

    manuscripts = list_manuscripts(user["id"])
    st.markdown("#### Your manuscripts")
    if not manuscripts:
        st.info("You have not added a manuscript yet.")
    for m in manuscripts:
        st.write(f"**{m['title']}**: {m['description']}")

    st.markdown("#### Commission a paid beta read (test mode)")
    readers = list_verified_readers()
    if not manuscripts:
        st.caption("Add a manuscript first.")
    elif not readers:
        st.caption("No verified readers yet. An admin must verify readers first.")
    else:
        with st.form("commission_form"):
            m_idx = st.selectbox(
                "Manuscript", range(len(manuscripts)), format_func=lambda i: manuscripts[i]["title"]
            )
            r_idx = st.selectbox(
                "Reader", range(len(readers)), format_func=lambda i: readers[i]["name"]
            )
            fee = st.number_input("Fee (₹, test mode)", min_value=100, max_value=10000, value=500, step=50)
            if st.form_submit_button("Commission and hold payment in escrow"):
                ok, msg = commission_beta_read(
                    user["id"], manuscripts[m_idx]["id"], readers[r_idx]["id"], int(fee)
                )
                flash(ok, msg)
                st.rerun()

    st.markdown("#### Your beta reads")
    for r in list_requests_for_author(user["id"]):
        with st.container(border=True):
            st.write(f"**{r['title']}**, read by **{r['reader_name']}**, fee ₹{r['price']}")
            st.caption(f"Status: {r['status']}  |  Escrow: {r['escrow_state']}")
            if r["plot"] is not None:
                st.write(
                    f"Scores (1-5): plot {r['plot']}, characters {r['characters']}, "
                    f"writing {r['writing']}, pacing {r['pacing']}, ending {r['ending']}"
                )
                if r["comments"]:
                    st.write(f"Comments: {r['comments']}")
            if r["status"] == "submitted":
                if st.button("Approve feedback and release payment", key=f"rel_{r['id']}"):
                    ok, msg = release_payment(user["id"], r["id"])
                    flash(ok, msg)
                    st.rerun()
            if r["status"] == "requested":
                if st.button("Cancel and refund", key=f"cancel_{r['id']}"):
                    ok, msg = cancel_and_refund(user["id"], r["id"])
                    flash(ok, msg)
                    st.rerun()


def page_beta_reading():
    user = st.session_state["user"]
    st.subheader("Beta reading")
    show_flash()

    if not is_verified(user["id"]):
        st.warning("Your account is waiting for admin verification.")
        return

    requests_ = list_requests_for_reader(user["id"])
    if not requests_:
        st.info("No beta-read requests yet.")
    for r in requests_:
        with st.container(border=True):
            st.write(f"**{r['title']}**, fee ₹{r['price']}")
            st.caption(f"Status: {r['status']}  |  Escrow: {r['escrow_state']}")
            st.write(r["description"])
            if r["status"] == "requested":
                with st.form(f"feedback_form_{r['id']}"):
                    scores = {
                        a: st.slider(a.capitalize(), 1, 5, 3, key=f"{a}_{r['id']}") for a in ASPECTS
                    }
                    comments = st.text_area("Comments", key=f"comments_{r['id']}")
                    if st.form_submit_button("Submit feedback"):
                        ok, msg = submit_feedback(user["id"], r["id"], scores, comments)
                        flash(ok, msg)
                        st.rerun()


def page_author_pool():
    user = st.session_state["user"]
    st.subheader("Author pool")
    show_flash()
    st.write(
        "Authors not yet selected by any publisher. Once you select an author, "
        "they disappear from every other publisher's list."
    )

    pool = list_available_authors()
    if not pool:
        st.info("No available authors right now.")
    for a in pool:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            if a["beta_score"] is None:
                score = "no completed beta reads yet"
            else:
                score = f"beta-read score {a['beta_score']:.1f}/5 from {a['beta_reads']} read(s)"
            c1.write(f"**{a['name']}**: {a['manuscripts']} manuscript(s) ({a['titles']}); {score}")
            if c2.button("Select", key=f"select_{a['id']}"):
                ok, msg = select_author(user["id"], a["id"])
                flash(ok, msg)
                st.rerun()

    st.markdown("#### My selected authors")
    mine = list_my_selections(user["id"])
    if mine:
        st.dataframe(pd.DataFrame(mine), width="stretch", hide_index=True)
    else:
        st.caption("You have not selected any author yet.")


# ---------------------------------------------------------------- routing by role
PAGES = {
    "Overview": page_overview,
    "Hidden Gems": page_hidden_gems,
    "Bestseller Predictor": page_predictor,
    "Author Pool": page_author_pool,
    "My Manuscripts": page_my_manuscripts,
    "Beta Reading": page_beta_reading,
    "Admin": page_admin,
    "Log in": page_login,
    "Register": page_register,
}

MENUS = {
    "publisher": ["Overview", "Hidden Gems", "Bestseller Predictor", "Author Pool"],
    "admin": ["Overview", "Hidden Gems", "Bestseller Predictor", "Admin"],
    "author": ["My Manuscripts"],
    "reader": ["Beta Reading"],
}

st.title("BookScope Intelligence")
user = st.session_state.get("user")

if user:
    st.sidebar.write(f"Signed in as **{user['name']}** ({user['role']})")
    if st.sidebar.button("Log out"):
        st.session_state.pop("user", None)
        st.rerun()
    menu = MENUS[user["role"]]
else:
    menu = ["Log in", "Register"]

choice = st.sidebar.radio("Page", menu)
PAGES[choice]()