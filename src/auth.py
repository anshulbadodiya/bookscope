import hashlib
import hmac
import os
import re
import sqlite3

from src.db import db

SELF_SERVE_ROLES = ["author", "reader", "publisher"]  # admin cannot be self-registered
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def hash_password(password):
    """Salted scrypt hash, stored as 'salt$hash' (both hex)."""
    salt = os.urandom(16)  # a fresh random salt for every password
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return salt.hex() + "$" + digest.hex()


def verify_password(password, stored):
    salt_hex, digest_hex = stored.split("$")
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return hmac.compare_digest(digest, bytes.fromhex(digest_hex))  # constant-time compare


def register_user(name, email, password, role):
    """Returns (ok, message)."""
    name, email = name.strip(), email.strip().lower()
    if not name:
        return False, "Please enter your name."
    if not EMAIL_PATTERN.fullmatch(email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if role not in SELF_SERVE_ROLES:
        return False, "Invalid role."

    verified = 0 if role == "reader" else 1  # readers must be approved by the admin
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users (email, name, password_hash, role, verified) "
                "VALUES (?, ?, ?, ?, ?)",
                (email, name, hash_password(password), role, verified),
            )
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    if role == "reader":
        return True, "Account created. An admin must verify you before you can review manuscripts."
    return True, "Account created. You can log in now."


def login_user(email, password):
    """Returns a user dict (without the hash) or None."""
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
    if row and verify_password(password, row["password_hash"]):
        return {k: row[k] for k in ("id", "name", "email", "role", "verified")}
    return None


def list_unverified_readers():
    with db() as conn:
        rows = conn.execute(
            "SELECT id, name, email, created_at FROM users "
            "WHERE role = 'reader' AND verified = 0 ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def verify_user(user_id):
    with db() as conn:
        conn.execute("UPDATE users SET verified = 1 WHERE id = ?", (user_id,))


def seed_admin():
    """Create the first admin if none exists. Demo password unless an env variable is set."""
    password = os.environ.get("BOOKSCOPE_ADMIN_PASSWORD", "admin123")
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM users WHERE role = 'admin'").fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO users (email, name, password_hash, role, verified) "
                "VALUES (?, ?, ?, 'admin', 1)",
                ("admin@bookscope.test", "Administrator", hash_password(password)),
            )