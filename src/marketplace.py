import sqlite3

from src.db import db

ASPECTS = ["plot", "characters", "writing", "pacing", "ending"]


class MarketplaceError(Exception):
    """A marketplace rule was broken (wrong role, not your item, wrong state...)."""


def _require_role(conn, user_id, role, must_be_verified=False):
    row = conn.execute("SELECT role, verified FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None or row["role"] != role:
        raise MarketplaceError(f"Only a {role} can do this.")
    if must_be_verified and not row["verified"]:
        raise MarketplaceError("This account has not been verified yet.")


def _rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def is_verified(user_id):
    """Fresh check from the database (the login session can be out of date)."""
    with db() as conn:
        row = conn.execute("SELECT verified FROM users WHERE id = ?", (user_id,)).fetchone()
    return bool(row and row["verified"])


# ------------------------------------------------------------------ author side
def add_manuscript(author_id, title, description):
    title, description = title.strip(), description.strip()
    if not title or len(description) < 20:
        return False, "Please enter a title and a description of at least 20 characters."
    try:
        with db() as conn:
            _require_role(conn, author_id, "author")
            conn.execute(
                "INSERT INTO manuscripts (author_id, title, description) VALUES (?, ?, ?)",
                (author_id, title, description),
            )
    except MarketplaceError as e:
        return False, str(e)
    return True, "Manuscript saved."


def list_manuscripts(author_id):
    with db() as conn:
        return _rows(
            conn,
            "SELECT id, title, description, created_at FROM manuscripts "
            "WHERE author_id = ? ORDER BY id DESC",
            (author_id,),
        )


def list_verified_readers():
    with db() as conn:
        return _rows(
            conn,
            "SELECT id, name FROM users WHERE role = 'reader' AND verified = 1 ORDER BY name",
        )


def commission_beta_read(author_id, manuscript_id, reader_id, price):
    """Create the request AND put the fee in escrow, as one all-or-nothing transaction."""
    if price <= 0:
        return False, "The fee must be positive."
    try:
        with db() as conn:
            _require_role(conn, author_id, "author")
            owned = conn.execute(
                "SELECT 1 FROM manuscripts WHERE id = ? AND author_id = ?",
                (manuscript_id, author_id),
            ).fetchone()
            if owned is None:
                raise MarketplaceError("That manuscript is not yours.")
            _require_role(conn, reader_id, "reader", must_be_verified=True)
            cur = conn.execute(
                "INSERT INTO beta_requests (manuscript_id, reader_id, price) VALUES (?, ?, ?)",
                (manuscript_id, reader_id, price),
            )
            conn.execute(
                "INSERT INTO escrow_ledger (beta_request_id, amount) VALUES (?, ?)",
                (cur.lastrowid, price),
            )
    except MarketplaceError as e:
        return False, str(e)
    except sqlite3.IntegrityError:
        return False, "You have already commissioned this reader for this manuscript."
    return True, f"Beta read commissioned. ₹{price} is held in escrow (test mode)."


def list_requests_for_author(author_id):
    with db() as conn:
        return _rows(
            conn,
            """
            SELECT br.id, m.title, u.name AS reader_name, br.price, br.status,
                   e.state AS escrow_state,
                   f.plot, f.characters, f.writing, f.pacing, f.ending, f.comments
            FROM beta_requests br
            JOIN manuscripts m   ON m.id = br.manuscript_id
            JOIN users u         ON u.id = br.reader_id
            JOIN escrow_ledger e ON e.beta_request_id = br.id
            LEFT JOIN beta_feedback f ON f.beta_request_id = br.id
            WHERE m.author_id = ?
            ORDER BY br.id DESC
            """,
            (author_id,),
        )


def _own_request(conn, author_id, request_id):
    _require_role(conn, author_id, "author")
    row = conn.execute(
        """
        SELECT br.status, e.state
        FROM beta_requests br
        JOIN manuscripts m   ON m.id = br.manuscript_id
        JOIN escrow_ledger e ON e.beta_request_id = br.id
        WHERE br.id = ? AND m.author_id = ?
        """,
        (request_id, author_id),
    ).fetchone()
    if row is None:
        raise MarketplaceError("That request is not yours.")
    return row


def release_payment(author_id, request_id):
    try:
        with db() as conn:
            row = _own_request(conn, author_id, request_id)
            if row["status"] != "submitted" or row["state"] != "held":
                raise MarketplaceError("Payment can only be released after feedback is submitted.")
            conn.execute(
                "UPDATE escrow_ledger SET state = 'released', updated_at = CURRENT_TIMESTAMP "
                "WHERE beta_request_id = ?",
                (request_id,),
            )
            conn.execute("UPDATE beta_requests SET status = 'completed' WHERE id = ?", (request_id,))
    except MarketplaceError as e:
        return False, str(e)
    return True, "Feedback approved. Payment released to the reader."


def cancel_and_refund(author_id, request_id):
    try:
        with db() as conn:
            row = _own_request(conn, author_id, request_id)
            if row["status"] != "requested" or row["state"] != "held":
                raise MarketplaceError("Only a request with no feedback yet can be cancelled.")
            conn.execute(
                "UPDATE escrow_ledger SET state = 'refunded', updated_at = CURRENT_TIMESTAMP "
                "WHERE beta_request_id = ?",
                (request_id,),
            )
            conn.execute("UPDATE beta_requests SET status = 'cancelled' WHERE id = ?", (request_id,))
    except MarketplaceError as e:
        return False, str(e)
    return True, "Request cancelled. The fee was refunded."


def author_selection(author_id):
    """Name of the publisher who selected this author, or None."""
    with db() as conn:
        row = conn.execute(
            "SELECT u.name FROM selections s JOIN users u ON u.id = s.publisher_id "
            "WHERE s.author_id = ?",
            (author_id,),
        ).fetchone()
    return row["name"] if row else None


# ------------------------------------------------------------------ reader side
def list_requests_for_reader(reader_id):
    with db() as conn:
        return _rows(
            conn,
            """
            SELECT br.id, m.title, m.description, br.price, br.status, e.state AS escrow_state
            FROM beta_requests br
            JOIN manuscripts m   ON m.id = br.manuscript_id
            JOIN escrow_ledger e ON e.beta_request_id = br.id
            WHERE br.reader_id = ?
            ORDER BY br.id DESC
            """,
            (reader_id,),
        )


def submit_feedback(reader_id, request_id, scores, comments):
    try:
        with db() as conn:
            _require_role(conn, reader_id, "reader", must_be_verified=True)
            row = conn.execute(
                "SELECT status FROM beta_requests WHERE id = ? AND reader_id = ?",
                (request_id, reader_id),
            ).fetchone()
            if row is None:
                raise MarketplaceError("That request is not assigned to you.")
            if row["status"] != "requested":
                raise MarketplaceError("Feedback was already submitted for this request.")
            conn.execute(
                "INSERT INTO beta_feedback "
                "(beta_request_id, plot, characters, writing, pacing, ending, comments) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (request_id, *[scores[a] for a in ASPECTS], comments.strip()),
            )
            conn.execute("UPDATE beta_requests SET status = 'submitted' WHERE id = ?", (request_id,))
    except MarketplaceError as e:
        return False, str(e)
    return True, "Feedback submitted. The author can now release your payment."


# ------------------------------------------------------------------ publisher side
def list_available_authors():
    """Authors with manuscripts who are NOT yet selected by any publisher."""
    with db() as conn:
        return _rows(
            conn,
            """
            SELECT u.id, u.name,
                   COUNT(DISTINCT m.id) AS manuscripts,
                   GROUP_CONCAT(DISTINCT m.title) AS titles,
                   COUNT(f.id) AS beta_reads,
                   AVG((f.plot + f.characters + f.writing + f.pacing + f.ending) / 5.0) AS beta_score
            FROM users u
            JOIN manuscripts m ON m.author_id = u.id
            LEFT JOIN beta_requests br ON br.manuscript_id = m.id AND br.status = 'completed'
            LEFT JOIN beta_feedback f ON f.beta_request_id = br.id
            LEFT JOIN selections s ON s.author_id = u.id
            WHERE u.role = 'author' AND s.id IS NULL
            GROUP BY u.id
            ORDER BY beta_score IS NULL, beta_score DESC
            """,
        )


def select_author(publisher_id, author_id):
    try:
        with db() as conn:
            _require_role(conn, publisher_id, "publisher")
            _require_role(conn, author_id, "author")
            conn.execute(
                "INSERT INTO selections (author_id, publisher_id) VALUES (?, ?)",
                (author_id, publisher_id),
            )
    except MarketplaceError as e:
        return False, str(e)
    except sqlite3.IntegrityError:
        return False, "Too late: another publisher has just selected this author."
    return True, "Author selected. They are now removed from all other publishers' lists."


def list_my_selections(publisher_id):
    with db() as conn:
        return _rows(
            conn,
            "SELECT u.name AS author, s.selected_at AS selected_at "
            "FROM selections s JOIN users u ON u.id = s.author_id "
            "WHERE s.publisher_id = ? ORDER BY s.id DESC",
            (publisher_id,),
        )


# ------------------------------------------------------------------ admin side
def list_escrow_ledger():
    with db() as conn:
        return _rows(
            conn,
            """
            SELECT e.id, m.title AS manuscript, ua.name AS author, ur.name AS reader,
                   e.amount, e.state, e.updated_at
            FROM escrow_ledger e
            JOIN beta_requests br ON br.id = e.beta_request_id
            JOIN manuscripts m    ON m.id = br.manuscript_id
            JOIN users ua         ON ua.id = m.author_id
            JOIN users ur         ON ur.id = br.reader_id
            ORDER BY e.id DESC
            """,
        )