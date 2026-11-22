from src.auth import register_user, verify_user
from src.db import db
from src.marketplace import (
    add_manuscript,
    commission_beta_read,
    list_manuscripts,
    list_requests_for_author,
    release_payment,
    submit_feedback,
)

DEMO_PASSWORD = "demo12345"
DEMO_USERS = [
    ("Demo Author", "author.demo@bookscope.test", "author"),
    ("Demo Author Two", "author2.demo@bookscope.test", "author"),
    ("Demo Reader", "reader.demo@bookscope.test", "reader"),
    ("Demo Publisher A", "pubA.demo@bookscope.test", "publisher"),
    ("Demo Publisher B", "pubB.demo@bookscope.test", "publisher"),
]


def _user_id(email):
    with db() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    return row["id"] if row else None


def load_demo_data():
    """Create sample accounts and one completed beta read. Safe to click twice."""
    if _user_id(DEMO_USERS[0][1]) is not None:
        return False, "Demo data is already loaded."

    for name, email, role in DEMO_USERS:
        register_user(name, email, DEMO_PASSWORD, role)

    author1 = _user_id("author.demo@bookscope.test")
    author2 = _user_id("author2.demo@bookscope.test")
    reader = _user_id("reader.demo@bookscope.test")
    verify_user(reader)  # readers normally need admin approval

    add_manuscript(
        author1,
        "The Lantern Keeper of Jabalpur",
        "A retired railway clerk discovers a box of unsent letters in an old station "
        "lamp room and sets out to deliver them forty years late.",
    )
    add_manuscript(
        author2,
        "Monsoon Protocol",
        "When a flood cuts off a mountain research station, a young hydrologist realises "
        "her data predicts a disaster no one believes.",
    )

    # Author One gets a full cycle: commission -> feedback -> payment released
    manuscript_id = list_manuscripts(author1)[0]["id"]
    commission_beta_read(author1, manuscript_id, reader, 500)
    request_id = list_requests_for_author(author1)[0]["id"]
    submit_feedback(
        reader,
        request_id,
        {"plot": 4, "characters": 5, "writing": 4, "pacing": 3, "ending": 4},
        "Warm, believable characters. The middle drags slightly, but the ending lands well.",
    )
    release_payment(author1, request_id)
    return True, "Demo data loaded. Log in with any demo account shown below."