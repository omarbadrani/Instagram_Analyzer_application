# -*- coding: utf-8 -*-
"""
PySide6 Instagram Analyzer - Enhanced Version
--------------------------
- Fetches account information (name, bio, followers, ...)
- Fetches comments from recent posts and classifies them (Positive / Negative / Question / Spam / Neutral)
- Runs locally using the instaloader library (unofficial).

⚠️ Warning:
- Using scraping may violate Instagram's terms and could lead to temporary bans. For production use, use the Instagram Graph API.
- Fetching comments for private accounts or some data may require login.
- To avoid 401/403 errors, always use login with a valid account, increase delays, and limit requests.
- Recommend running `pip install --upgrade instaloader` or `pip install --pre instaloader` for latest alpha version.

Requirements:
pip install pyside6 instaloader pandas textblob

Run:
python instagram_ai_analyzer.py
"""
import sys
import re
import traceback
import time
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QFileDialog,
    QSpinBox, QCheckBox, QHBoxLayout, QHeaderView, QVBoxLayout, QProgressBar,
    QTabWidget, QGroupBox, QMessageBox
)
from PySide6.QtGui import QFont, QColor

# Prepare imports here so the app doesn't crash if the library is not installed
try:
    import instaloader
except ImportError:
    instaloader = None
    print("⚠️ instaloader not installed. Run: pip install instaloader")

try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠️ pandas not installed. Run: pip install pandas")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("⚠️ TextBlob not installed. Will use basic analysis only. For English text: pip install textblob")

# ---------- AI: Advanced Comment Classification (Rules + Sentiment Analysis) ---------- #
POS_WORDS = set(
    ["good", "great", "awesome", "love", "nice", "amazing", "cool", "wow", "perfect", "beautiful", "like", "thanks",
     "thank you", "congrats", "best", "bravo",
     "جميل", "حلو", "رايع", "رائع", "ممتاز", "باهي", "برشا بهي", "يعطيك الصحة", "سلمت", "تحفة", "خطير", "يعجبني",
     "مبدع", "مميز", "ابدااع", "يعطيكم الصحة",
     "ماشاء الله", "ما شاء الله", "تبارك الله", "واو", "جميلة", "رائعة", "ممتازة"])

NEG_WORDS = set(
    ["bad", "hate", "worst", "ugly", "terrible", "awful", "dislike", "boring", "fake", "trash", "disappointing",
     "سيء", "سئ", "نكره", "نكرة", "فاشل", "خايب", "خيب", "مو حلو", "مش حلو", "رديء", "ضعيف", "كريه", "تعبان", "نصب",
     "احتيال", "كذب",
     "مقرف", "مقرفة", "سيئة", "سىء", "خايب", "خايبية"])

SPAM_PATTERNS = [
    r"http[s]?://", "www\\.",
    r"\b(win|free|giveaway|promo|discount|deal|offer|prize|competition|contest)\b",
    r"\b(follow\s*me|dm\s*me|check\s*my\s*bio|link\s*in\s*bio|click\s*link|visit\s*profile)\b",
    r"ربح|مجانا|تخفيض|خصم|تابعني|اتصل بنا|اذهب للبروفايل|زيارة الموقع|عرض خاص"
]

QUESTION_CUES = ["?", "؟", "كيف", "ليش", "متى", "وين", "أين", "هل", "شنو", "شكون", "علاش",
                 "Pourquoi", "how", "why", "when", "where", "what", "which", "who", "whom",
                 "هل", "كم", "ماذا", "لماذا", "أين", "كيف", "متى"]

HEART_EMOJIS = "❤♥💖💗💓💝💘💞💕💟😍😘😻💯🔥"
POSITIVE_EMOJIS = HEART_EMOJIS + "😊😂🤣😁👍👏🎉✨🌟⭐💪🤩"
NEGATIVE_EMOJIS = "👎😠😡🤬😒🙄💔😞😢😭🤢🤮"


@dataclass
class CommentRow:
    post_date: datetime
    username: str
    text: str
    label: str
    score: float
    sentiment: str  # Positive/Negative/Neutral


def clean_text(s: str) -> str:
    """Clean text from extra spaces and unnecessary characters"""
    return re.sub(r"\s+", " ", s).strip()


def count_any(s: str, vocab: set) -> int:
    """Count words from vocabulary in text"""
    s_low = s.lower()
    return sum(1 for w in vocab if w in s_low)


def count_emojis(s: str, emoji_set: str) -> int:
    """Count emojis in text"""
    return sum(1 for char in s if char in emoji_set)


def looks_like_spam(s: str) -> bool:
    """Detect spam comments"""
    s_low = s.lower()

    # Check patterns
    for pat in SPAM_PATTERNS:
        if re.search(pat, s_low, flags=re.IGNORECASE):
            return True

    # Detect excessive hashtags or mentions
    if s_low.count("#") >= 5 or s_low.count("@") >= 5:
        return True

    # Detect excessive repetition
    if re.search(r"(.)\1{4,}", s_low):
        return True

    # Spam keywords
    spam_keywords = ["follow", "like", "comment", "share", "tag", "win", "free", "giveaway",
                     "تابع", "لايك", "كومنت", "شير", "تاق", "ربح", "مجاني", "هبة"]
    if any(keyword in s_low for keyword in spam_keywords):
        return True

    return False


def is_mostly_english(text: str) -> bool:
    """Check if text is mostly English (more than 50% English characters)"""
    english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    total_chars = sum(1 for c in text if c.isalpha())
    return total_chars > 0 and (english_chars / total_chars) > 0.5


def analyze_with_textblob(text: str) -> Tuple[str, float]:
    """Sentiment analysis using TextBlob (for English text)"""
    if not HAS_TEXTBLOB or not text.strip():
        return ("Neutral", 0.0)

    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity

        if polarity > 0.1:
            return ("Positive", min(1.0, polarity))
        elif polarity < -0.1:
            return ("Negative", max(-1.0, polarity))
        else:
            return ("Neutral", 0.0)
    except:
        return ("Neutral", 0.0)


def classify_comment(text: str) -> Tuple[str, float, str]:
    """Classify comment with improved algorithm"""
    t = clean_text(text)
    if not t:
        return ("Neutral", 0.0, "Neutral")

    # Detect spam first
    if looks_like_spam(t):
        return ("Spam", -0.5, "Negative")

    # Detect questions
    if any(cue in t for cue in QUESTION_CUES):
        return ("Question", 0.0, "Neutral")

    # Use TextBlob for mostly English text
    if is_mostly_english(t) and len(t) > 3:
        label, score = analyze_with_textblob(t)
        return (label, score, label)

    # Analysis for Arabic and others
    pos_count = count_any(t, POS_WORDS)
    neg_count = count_any(t, NEG_WORDS)

    # Count emojis
    pos_count += count_emojis(t, POSITIVE_EMOJIS) * 0.7  # Lower weight for emojis
    neg_count += count_emojis(t, NEGATIVE_EMOJIS) * 0.7

    # Determine basic sentiment
    if pos_count > neg_count:
        sentiment = "Positive"
    elif neg_count > pos_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Calculate score
    total_words = max(1, len(t.split()))
    score = (pos_count - neg_count) / total_words
    score = max(-1.0, min(1.0, score))

    # Final classification
    if abs(score) > 0.3:
        label = "Positive" if score > 0 else "Negative"
    elif abs(score) > 0.1:
        label = sentiment
    else:
        label = "Neutral"

    return (label, score, sentiment)


# -------------------- Workers -------------------- #
class ProfileWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, username: str, do_login: bool, login_user: str, login_pass: str, parent=None):
        super().__init__(parent)
        self.username = username
        self.do_login = do_login
        self.login_user = login_user
        self.login_pass = login_pass

    def run(self):
        try:
            if instaloader is None:
                raise RuntimeError("instaloader not installed. Run: pip install instaloader")

            self.progress.emit("Initializing Instaloader...")
            # Set custom user agent to mimic browser and increase attempts
            L = instaloader.Instaloader(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
                max_connection_attempts=5
            )
            L.context.sleep_time = 15  # Increased delay to avoid rate limits

            if self.do_login and self.login_user:
                self.progress.emit("Attempting to load existing session or login...")
                try:
                    L.load_session_from_file(self.login_user)
                    self.progress.emit("Loaded existing session successfully")
                except FileNotFoundError:
                    if self.login_pass:
                        try:
                            L.login(self.login_user, self.login_pass)
                            self.progress.emit("Login successful")
                        except Exception as e:
                            raise RuntimeError(f"Login failed: {e}. Try manual login via CLI: instaloader --login={self.login_user}")
                    else:
                        raise RuntimeError("No password provided and no session file found.")

            self.progress.emit("Fetching profile data...")
            profile = instaloader.Profile.from_username(L.context, self.username)

            data = {
                "username": profile.username,
                "full_name": profile.full_name,
                "biography": profile.biography,
                "followers": profile.followers,
                "followees": profile.followees,
                "mediacount": profile.mediacount,
                "is_private": profile.is_private,
                "is_verified": profile.is_verified,
                "external_url": getattr(profile, "external_url", ""),
                "profile_pic": profile.profile_pic_url
            }

            self.progress.emit("Profile data fetched successfully")
            self.finished_ok.emit(data)

        except instaloader.exceptions.ProfileNotExistsException:
            self.failed.emit("Profile does not exist or is unavailable")
        except instaloader.exceptions.ConnectionException as e:
            self.failed.emit(f"Connection failed (likely rate limit). Try login, wait, or VPN: {e}")
        except Exception as e:
            self.failed.emit(f"Profile error: {e}\n{traceback.format_exc()}")


class CommentsWorker(QThread):
    finished_ok = Signal(list, dict)  # Comments + stats
    failed = Signal(str)
    progress = Signal(str, int)  # Message, percentage

    def __init__(self, username: str, max_posts: int, max_comments_per_post: int,
                 do_login: bool, login_user: str, login_pass: str, parent=None):
        super().__init__(parent)
        self.username = username
        self.max_posts = max_posts
        self.max_comments_per_post = max_comments_per_post
        self.do_login = do_login
        self.login_user = login_user
        self.login_pass = login_pass
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            if instaloader is None:
                raise RuntimeError("instaloader not installed. Run: pip install instaloader")

            self.progress.emit("Initializing Instaloader...", 5)
            # Set custom user agent and increase attempts
            L = instaloader.Instaloader(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
                max_connection_attempts=5
            )
            L.context.sleep_time = 15  # Increased delay to avoid rate limits

            if self.do_login and self.login_user:
                self.progress.emit("Attempting to load existing session or login...", 10)
                try:
                    L.load_session_from_file(self.login_user)
                    self.progress.emit("Loaded existing session successfully", 15)
                except FileNotFoundError:
                    if self.login_pass:
                        try:
                            L.login(self.login_user, self.login_pass)
                            self.progress.emit("Login successful", 15)
                        except Exception as e:
                            raise RuntimeError(f"Login failed: {e}. Try manual login via CLI: instaloader --login={self.login_user}")
                    else:
                        raise RuntimeError("No password provided and no session file found.")

            self.progress.emit("Searching for profile...", 20)
            profile = instaloader.Profile.from_username(L.context, self.username)

            if profile.is_private:
                self.progress.emit("⚠️ Private profile. May not access posts", 25)

            self.progress.emit("Fetching posts...", 30)
            posts_iter = profile.get_posts()
            posts = []
            for _ in range(self.max_posts):
                if not self.is_running:
                    self.progress.emit("Process stopped", 0)
                    return
                try:
                    posts.append(next(posts_iter))
                    time.sleep(3)  # Increased delay between post fetches
                except StopIteration:
                    break
                except instaloader.exceptions.ConnectionException as e:
                    self.progress.emit(f"⚠️ Connection error during post fetch: {e}. Retrying after delay...", 30)
                    time.sleep(60)  # Longer wait on error

            if not posts:
                self.progress.emit("⚠️ No public posts", 40)

            rows: List[CommentRow] = []
            total_posts = len(posts)

            for i, post in enumerate(posts):
                if not self.is_running:
                    return

                progress = 40 + (i * 50 // total_posts) if total_posts > 0 else 40
                self.progress.emit(f"Processing post {i + 1}/{total_posts}...", progress)

                try:
                    comments_iter = post.get_comments()
                    comments = []
                    for j in range(self.max_comments_per_post if self.max_comments_per_post > 0 else 999999):
                        try:
                            comments.append(next(comments_iter))
                            time.sleep(1.5)  # Increased delay between comment fetches
                        except StopIteration:
                            break
                        except instaloader.exceptions.ConnectionException as e:
                            self.progress.emit(f"⚠️ Connection error during comment fetch {j+1}: {e}. Retrying after delay...", progress)
                            time.sleep(60)
                except Exception as e:
                    self.progress.emit(f"⚠️ Failed to fetch comments for post {i + 1}: {e}", progress)
                    comments = []

                for c in comments:
                    if not self.is_running:
                        return

                    text = getattr(c, 'text', '') or ''
                    user = getattr(c.owner, 'username', '') if hasattr(c, 'owner') and c.owner else ''
                    label, score, sentiment = classify_comment(text)

                    rows.append(CommentRow(
                        post.date_local if hasattr(post, 'date_local') else post.date,
                        user, text, label, score, sentiment
                    ))

            # Calculate stats
            stats = self.calculate_stats(rows)
            self.progress.emit("All comments analyzed successfully", 100)
            self.finished_ok.emit(rows, stats)

        except instaloader.exceptions.ProfileNotExistsException:
            self.failed.emit("Profile does not exist or is unavailable")
        except instaloader.exceptions.ConnectionException as e:
            self.failed.emit(f"Connection failed (likely rate limit). Try login, wait, or VPN: {e}")
        except Exception as e:
            self.failed.emit(f"Comments error: {e}\n{traceback.format_exc()}")

    def calculate_stats(self, rows: List[CommentRow]) -> Dict[str, Any]:
        """Calculate comment statistics"""
        if not rows:
            return {}

        total = len(rows)
        stats = {
            "total": total,
            "label_positive": 0,
            "label_negative": 0,
            "label_neutral": 0,
            "label_question": 0,
            "label_spam": 0,
            "sent_positive": 0,
            "sent_negative": 0,
            "sent_neutral": 0,
            "avg_score": 0
        }

        for row in rows:
            # Accumulate score
            stats["avg_score"] += row.score

            # Count labels
            if row.label == "Positive":
                stats["label_positive"] += 1
            elif row.label == "Negative":
                stats["label_negative"] += 1
            elif row.label == "Neutral":
                stats["label_neutral"] += 1
            elif row.label == "Question":
                stats["label_question"] += 1
            elif row.label == "Spam":
                stats["label_spam"] += 1

            # Count sentiments
            if row.sentiment == "Positive":
                stats["sent_positive"] += 1
            elif row.sentiment == "Negative":
                stats["sent_negative"] += 1
            elif row.sentiment == "Neutral":
                stats["sent_neutral"] += 1

        stats["avg_score"] = stats["avg_score"] / total if total > 0 else 0

        return stats


# -------------------- Enhanced UI -------------------- #
class AnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Instagram AI Analyzer - PySide6 (Enhanced)")
        self.resize(1200, 800)
        self.setStyleSheet(self.get_stylesheet())
        self._build_ui()
        self.rows: List[CommentRow] = []
        self.stats: Dict[str, Any] = {}
        self.current_worker = None

    def get_stylesheet(self):
        return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QLabel {
            color: #333;
            font-weight: bold;
        }
        QPushButton {
            background-color: #4a86e8;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3a76d8;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QLineEdit, QTextEdit, QSpinBox {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 6px;
            background-color: white;
        }
        QTableWidget {
            gridline-color: #ddd;
            background-color: white;
            alternate-background-color: #f9f9f9;
        }
        QHeaderView::section {
            background-color: #eaeaea;
            padding: 6px;
            border: 1px solid #ddd;
            font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #ddd;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #eaeaea;
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #4a86e8;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        """

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        self.tabs.addTab(analysis_tab, "🔍 Analysis")

        # Account settings group
        account_group = QGroupBox("Account Settings")
        account_layout = QGridLayout(account_group)
        account_layout.setContentsMargins(10, 15, 10, 10)
        account_layout.setHorizontalSpacing(10)
        account_layout.setVerticalSpacing(8)

        r = 0
        account_layout.addWidget(QLabel("👤 Instagram Username:"), r, 0)
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Example: cristiano")
        account_layout.addWidget(self.username_edit, r, 1)
        account_layout.addWidget(QLabel("📥 Number of Posts (for comments):"), r, 2)
        self.posts_spin = QSpinBox()
        self.posts_spin.setRange(1, 100)
        self.posts_spin.setValue(5)
        account_layout.addWidget(self.posts_spin, r, 3)
        account_layout.addWidget(QLabel("💬 Comments Limit per Post:"), r, 4)
        self.comments_spin = QSpinBox()
        self.comments_spin.setRange(0, 500)
        self.comments_spin.setValue(50)
        account_layout.addWidget(self.comments_spin, r, 5)

        r += 1
        self.login_chk = QCheckBox("Login (recommended to avoid errors)")
        account_layout.addWidget(self.login_chk, r, 0)
        self.ig_user = QLineEdit()
        self.ig_user.setPlaceholderText("Instagram Username")
        self.ig_pass = QLineEdit()
        self.ig_pass.setPlaceholderText("Instagram Password")
        self.ig_pass.setEchoMode(QLineEdit.Password)
        account_layout.addWidget(self.ig_user, r, 1, 1, 2)
        account_layout.addWidget(self.ig_pass, r, 3, 1, 2)

        analysis_layout.addWidget(account_group)

        # Buttons group
        button_group = QWidget()
        button_layout = QHBoxLayout(button_group)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_profile = QPushButton("Fetch Profile Info")
        self.btn_comments = QPushButton("Fetch Comments + Analyze")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setEnabled(False)

        button_layout.addWidget(self.btn_profile)
        button_layout.addWidget(self.btn_comments)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_export)
        button_layout.addStretch()

        analysis_layout.addWidget(button_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        analysis_layout.addWidget(self.progress_bar)

        # Results group
        results_group = QGroupBox("Results")
        results_layout = QHBoxLayout(results_group)

        # Profile info
        self.profile_box = QTextEdit()
        self.profile_box.setReadOnly(True)
        self.profile_box.setPlaceholderText("Profile info will appear here…")
        results_layout.addWidget(self.profile_box, 1)

        # Log and stats
        right_panel = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(200)
        self.log_box.setPlaceholderText("Log / Errors…")

        # Quick stats
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMaximumHeight(150)
        self.stats_box.setPlaceholderText("Stats will appear here…")

        right_panel.addWidget(QLabel("📊 Stats:"))
        right_panel.addWidget(self.stats_box)
        right_panel.addWidget(QLabel("📋 Log:"))
        right_panel.addWidget(self.log_box)

        results_layout.addLayout(right_panel, 1)
        analysis_layout.addWidget(results_group)

        # Comments table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Post Date", "Comment User", "Comment Text", "Label", "Score", "Sentiment"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        analysis_layout.addWidget(self.table)

        # Stats tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_detail = QTextEdit()
        self.stats_detail.setReadOnly(True)
        self.stats_detail.setPlaceholderText("Detailed stats will appear here…")
        stats_layout.addWidget(self.stats_detail)
        self.tabs.addTab(stats_tab, "📊 Detailed Stats")

        self.setCentralWidget(central)

        # Connect signals
        self.btn_profile.clicked.connect(self.on_fetch_profile)
        self.btn_comments.clicked.connect(self.on_fetch_comments)
        self.btn_stop.clicked.connect(self.on_stop_process)
        self.btn_export.clicked.connect(self.on_export_csv)
        self.login_chk.stateChanged.connect(self.on_login_changed)

        # Disable login fields initially
        self.ig_user.setEnabled(False)
        self.ig_pass.setEnabled(False)

    def on_login_changed(self, state):
        enabled = state == Qt.Checked
        self.ig_user.setEnabled(enabled)
        self.ig_pass.setEnabled(enabled)

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")
        # Scroll to bottom
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def update_progress(self, msg: str, value: int = None):
        if value is not None:
            self.progress_bar.setValue(value)
        if msg:
            self.log(msg)

    def on_fetch_profile(self):
        username = self.username_edit.text().strip()
        if not username:
            QMessageBox.warning(self, "Warning", "Please enter the username.")
            return

        if not self.login_chk.isChecked():
            reply = QMessageBox.question(self, "Warning", "Login is highly recommended to avoid 401/403 errors. Continue without login?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.log("⏳ Fetching profile info…")
        self.btn_profile.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.pworker = ProfileWorker(
            username,
            self.login_chk.isChecked(),
            self.ig_user.text().strip(),
            self.ig_pass.text()
        )
        self.pworker.finished_ok.connect(self.on_profile_ok)
        self.pworker.failed.connect(self.on_worker_failed)
        self.pworker.progress.connect(self.update_progress)
        self.current_worker = self.pworker
        self.pworker.start()

    def on_profile_ok(self, data: dict):
        self.btn_profile.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.current_worker = None

        pretty = [
            f"👤 Username: {data.get('username', '')}",
            f"📛 Full Name: {data.get('full_name', '')}",
            f"🔗 External URL: {data.get('external_url', '')}",
            f"📝 Bio: {data.get('biography', '')}",
            f"👥 Followers: {data.get('followers', '')} | Following: {data.get('followees', '')}",
            f"📸 Posts: {data.get('mediacount', '')}",
            f"🔒 Private? {'Yes' if data.get('is_private') else 'No'} | ✅ Verified? {'Yes' if data.get('is_verified') else 'No'}"
        ]
        self.profile_box.setPlainText("\n".join(pretty))
        self.log("✅ Profile info fetched.")

    def on_fetch_comments(self):
        username = self.username_edit.text().strip()
        if not username:
            QMessageBox.warning(self, "Warning", "Please enter the username.")
            return

        if not self.login_chk.isChecked():
            reply = QMessageBox.question(self, "Warning", "Login is highly recommended to avoid 401/403 errors. Continue without login?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.log("⏳ Fetching and analyzing comments…")
        self.btn_comments.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)
        self.rows = []
        self.stats = {}

        self.cworker = CommentsWorker(
            username,
            self.posts_spin.value(),
            self.comments_spin.value(),
            self.login_chk.isChecked(),
            self.ig_user.text().strip(),
            self.ig_pass.text()
        )
        self.cworker.finished_ok.connect(self.on_comments_ok)
        self.cworker.failed.connect(self.on_worker_failed)
        self.cworker.progress.connect(lambda msg, value: self.update_progress(msg, value))
        self.current_worker = self.cworker
        self.cworker.start()

    def on_stop_process(self):
        if self.current_worker and hasattr(self.current_worker, 'stop'):
            self.current_worker.stop()
            self.log("⏹️ Process stop requested...")
            self.btn_stop.setEnabled(False)

    def on_comments_ok(self, rows: List[CommentRow], stats: Dict[str, Any]):
        self.btn_comments.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.current_worker = None

        self.rows = rows
        self.stats = stats

        # Update table
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(row.post_date)))
            self.table.setItem(i, 1, QTableWidgetItem(row.username))
            self.table.setItem(i, 2, QTableWidgetItem(row.text))

            # Color cells based on label
            label_item = QTableWidgetItem(row.label)
            if row.label == "Positive":
                label_item.setBackground(QColor(200, 255, 200))
            elif row.label == "Negative":
                label_item.setBackground(QColor(255, 200, 200))
            elif row.label == "Question":
                label_item.setBackground(QColor(200, 200, 255))
            elif row.label == "Spam":
                label_item.setBackground(QColor(255, 220, 200))
            self.table.setItem(i, 3, label_item)

            score_item = QTableWidgetItem(f"{row.score:.2f}")
            score_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(i, 4, score_item)

            sentiment_item = QTableWidgetItem(row.sentiment)
            if row.sentiment == "Positive":
                sentiment_item.setBackground(QColor(200, 255, 200))
            elif row.sentiment == "Negative":
                sentiment_item.setBackground(QColor(255, 200, 200))
            self.table.setItem(i, 5, sentiment_item)

        self.btn_export.setEnabled(len(rows) > 0)
        self.log(f"✅ Analysis complete, comments: {len(rows)}")

        # Update stats display
        self.update_stats_display()

    def update_stats_display(self):
        if not self.stats:
            return

        total = self.stats.get("total", 0)
        if total == 0:
            self.stats_box.setPlainText("No comments to analyze")
            return

        # Quick stats
        stats_text = [
            f"📊 Total Comments: {total}",
            f"✅ Positive: {self.stats.get('label_positive', 0)} ({self.stats.get('label_positive', 0) / total * 100:.1f}%)",
            f"❌ Negative: {self.stats.get('label_negative', 0)} ({self.stats.get('label_negative', 0) / total * 100:.1f}%)",
            f"📝 Neutral: {self.stats.get('label_neutral', 0)} ({self.stats.get('label_neutral', 0) / total * 100:.1f}%)",
            f"❓ Questions: {self.stats.get('label_question', 0)} ({self.stats.get('label_question', 0) / total * 100:.1f}%)",
            f"🚫 Spam: {self.stats.get('label_spam', 0)} ({self.stats.get('label_spam', 0) / total * 100:.1f}%)",
            f"⭐ Average Score: {self.stats.get('avg_score', 0):.2f}"
        ]
        self.stats_box.setPlainText("\n".join(stats_text))

        # Detailed stats
        detail_text = ["📈 Detailed Label Analysis:", "=" * 40]
        detail_text.append(f"Positive: {self.stats.get('label_positive', 0)} ({self.stats.get('label_positive', 0) / total * 100:.1f}%)")
        detail_text.append(f"Negative: {self.stats.get('label_negative', 0)} ({self.stats.get('label_negative', 0) / total * 100:.1f}%)")
        detail_text.append(f"Neutral: {self.stats.get('label_neutral', 0)} ({self.stats.get('label_neutral', 0) / total * 100:.1f}%)")
        detail_text.append(f"Question: {self.stats.get('label_question', 0)} ({self.stats.get('label_question', 0) / total * 100:.1f}%)")
        detail_text.append(f"Spam: {self.stats.get('label_spam', 0)} ({self.stats.get('label_spam', 0) / total * 100:.1f}%)")

        detail_text.extend([
            "",
            "📋 Sentiment Summary:",
            f"Positive: {self.stats.get('sent_positive', 0)} comments",
            f"Negative: {self.stats.get('sent_negative', 0)} comments",
            f"Neutral: {self.stats.get('sent_neutral', 0)} comments",
            "",
            f"📊 Average Comment Score: {self.stats.get('avg_score', 0):.2f}",
            f"💡 Overall Opinion: {'Positive' if self.stats.get('avg_score', 0) > 0.1 else 'Negative' if self.stats.get('avg_score', 0) < -0.1 else 'Neutral'}"
        ])

        self.stats_detail.setPlainText("\n".join(detail_text))

    def on_worker_failed(self, msg: str):
        self.btn_profile.setEnabled(True)
        self.btn_comments.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.current_worker = None

        # Show error message in dialog
        error_msg = msg.split("\n")[0]  # Take first line for display
        QMessageBox.critical(self, "Error", error_msg + "\n\nSuggestions: Update instaloader, use VPN, or manual browser cookie import for session.")
        self.log(f"❌ {msg}")

    def on_export_csv(self):
        if not self.rows:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return

        if pd is None:
            QMessageBox.warning(self, "Warning", "Need pandas for export: pip install pandas")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV File",
            f"instagram_analysis_{self.username_edit.text().strip()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        try:
            df = pd.DataFrame([{
                "post_date": r.post_date,
                "comment_user": r.username,
                "comment_text": r.text,
                "label": r.label,
                "score": r.score,
                "sentiment": r.sentiment
            } for r in self.rows])

            df.to_csv(path, index=False, encoding='utf-8-sig')
            self.log(f"💾 File saved: {path}")
            QMessageBox.information(self, "Export Done", f"Data saved to: {path}")

        except Exception as e:
            self.log(f"❌ Save failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def closeEvent(self, event):
        """Ensure any running processes are stopped on app close"""
        if self.current_worker and self.current_worker.isRunning():
            if hasattr(self.current_worker, 'stop'):
                self.current_worker.stop()
            self.current_worker.wait(5000)  # Wait up to 5 seconds
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Instagram AI Analyzer")
    app.setApplicationVersion("2.2")  # Updated version with fixes

    # Load suitable font for Arabic if available
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)

    w = AnalyzerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()