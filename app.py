import os
os.environ['MPLBACKEND'] = 'Agg'

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import io
import base64
import secrets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# ── Flask-Login setup ─────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access SpendIQ.'

# ── Database config ──────────────────────────────────────────
DB_CONFIG = {
    'host':     os.environ.get('MYSQLHOST',     'localhost'),
    'port':     int(os.environ.get('MYSQLPORT', 3306)),
    'database': os.environ.get('MYSQLDATABASE', 'expense_tracker'),
    'user':     os.environ.get('MYSQLUSER',     'root'),
    'password': os.environ.get('MYSQLPASSWORD', 'ayush'),  # set your local password here
}


# ── User model ───────────────────────────────────────────────

class User(UserMixin):
    def __init__(self, id, username, email):
        self.id       = id
        self.username = username
        self.email    = email


@login_manager.user_loader
def load_user(user_id):
    connection = get_db_connection()
    if not connection:
        return None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        cursor.close()
        connection.close()
        if row:
            return User(row['id'], row['username'], row['email'])
        return None
    except Error:
        return None


# ── Database helpers ─────────────────────────────────────────

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None


def create_tables_if_not_exist():
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INT AUTO_INCREMENT PRIMARY KEY,
                    username   VARCHAR(50)  UNIQUE NOT NULL,
                    email      VARCHAR(120) UNIQUE NOT NULL,
                    password   VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Expenses table — with user_id foreign key
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    user_id     INT NOT NULL,
                    amount      DECIMAL(10, 2) NOT NULL,
                    category    VARCHAR(50)    NOT NULL,
                    description VARCHAR(255),
                    date        DATE NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            connection.commit()
            cursor.close()
            connection.close()
        except Error as e:
            print(f"Table creation error: {e}")


# ── Chart helpers ─────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='none', edgecolor='none', dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def make_weekly_chart(week_data):
    dates = list(week_data.keys())
    amounts = list(week_data.values())
    date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    x_labels = [d.strftime('%a\n%d %b') for d in date_objects]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    sns.lineplot(x=x_labels, y=amounts, ax=ax,
                 color='#ffd700', linewidth=2.5, marker='o',
                 markersize=8, markerfacecolor='#ffd700', markeredgecolor='white',
                 markeredgewidth=1.5)
    ax.fill_between(x_labels, amounts, color='#ffd700', alpha=0.18)
    ax.set_facecolor('none')
    ax.tick_params(colors='white', labelsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v:.0f}'))
    ax.set_xlabel('', color='white')
    ax.set_ylabel('Amount (₹)', color='white', fontsize=9)
    ax.spines['bottom'].set_color('rgba(255,255,255,0.3)')
    ax.spines['left'].set_color('rgba(255,255,255,0.3)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='white', alpha=0.1, linestyle='--')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')
    plt.tight_layout()
    return fig_to_base64(fig)


def make_monthly_chart(categories, amounts):
    if not categories:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.text(0.5, 0.5, 'No expenses this month', color='white',
                ha='center', va='center', fontsize=13, transform=ax.transAxes)
        ax.axis('off')
        plt.tight_layout()
        return fig_to_base64(fig)

    palette = sns.color_palette("husl", len(categories))
    fig, ax = plt.subplots(figsize=(7, max(3.8, len(categories) * 0.55)))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    bars = ax.barh(categories, amounts, color=palette, edgecolor='white',
                   linewidth=0.5, height=0.55)
    for bar, amt in zip(bars, amounts):
        ax.text(bar.get_width() + max(amounts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'₹{amt:.0f}', va='center', color='white', fontsize=9, fontweight='bold')

    ax.set_facecolor('none')
    ax.tick_params(colors='white', labelsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'₹{v:.0f}'))
    ax.spines['bottom'].set_color('rgba(255,255,255,0.3)')
    ax.spines['left'].set_color('rgba(255,255,255,0.3)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color='white', alpha=0.1, linestyle='--')
    ax.set_xlim(0, max(amounts) * 1.25)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')
    plt.tight_layout()
    return fig_to_base64(fig)


# ── ML prediction ─────────────────────────────────────────────

def predict_next_week(connection, user_id):
    try:
        cursor = connection.cursor()
        end_date   = datetime.now().date()
        start_date = end_date - timedelta(days=29)
        cursor.execute("""
            SELECT DATE(date) as d, SUM(amount) as total
            FROM expenses
            WHERE date BETWEEN %s AND %s AND user_id = %s
            GROUP BY DATE(date)
            ORDER BY d
        """, (start_date, end_date, user_id))
        rows = cursor.fetchall()
        cursor.close()

        if len(rows) < 5:
            return None

        df = pd.DataFrame(rows, columns=['date', 'total'])
        df['day_num'] = (df['date'] - df['date'].min()).apply(lambda x: x.days)
        X = df[['day_num']].values
        y = df['total'].values.astype(float)

        model = LinearRegression()
        model.fit(X, y)
        next_days = np.array([[i] for i in range(30, 37)])
        predicted_weekly = max(0.0, float(model.predict(next_days).sum()))
        return round(predicted_weekly, 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Auth routes
# ──────────────────────────────────────────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        # Basic validation
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('register.html')
        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        connection = get_db_connection()
        if not connection:
            flash('Database error. Please try again.', 'error')
            return render_template('register.html')

        try:
            cursor = connection.cursor(dictionary=True)

            # Check if username or email already taken
            cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s",
                           (username, email))
            existing = cursor.fetchone()
            if existing:
                flash('Username or email already in use.', 'error')
                cursor.close()
                connection.close()
                return render_template('register.html')

            hashed_password = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, hashed_password)
            )
            connection.commit()
            new_id = cursor.lastrowid
            cursor.close()
            connection.close()

            user = User(new_id, username, email)
            login_user(user)
            flash(f'Welcome to SpendIQ, {username}! 🎉', 'success')
            return redirect(url_for('home'))

        except Error as e:
            flash('Registration failed. Please try again.', 'error')
            print(f"Register error: {e}")
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        if not username or not password:
            flash('Please enter username and password.', 'error')
            return render_template('login.html')

        connection = get_db_connection()
        if not connection:
            flash('Database error. Please try again.', 'error')
            return render_template('login.html')

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            row = cursor.fetchone()
            cursor.close()
            connection.close()

            if row and check_password_hash(row['password'], password):
                user = User(row['id'], row['username'], row['email'])
                login_user(user, remember=remember)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('home'))
            else:
                flash('Invalid username or password.', 'error')
                return render_template('login.html')

        except Error as e:
            flash('Login failed. Please try again.', 'error')
            print(f"Login error: {e}")
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ──────────────────────────────────────────────────────────────
# Main app routes (all require login)
# ──────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def home():
    return render_template('index.html', username=current_user.username)


@app.route('/add_expense', methods=['POST'])
@login_required
def add_expense():
    try:
        data = request.get_json()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO expenses (user_id, amount, category, description, date) VALUES (%s, %s, %s, %s, %s)",
            (current_user.id, data['amount'], data['category'], data['description'], data['date'])
        )
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'success': True, 'message': 'Expense added successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get_expenses')
@login_required
def get_expenses():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify([])
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM expenses WHERE user_id = %s ORDER BY date DESC",
                       (current_user.id,))
        expenses = cursor.fetchall()
        for expense in expenses:
            expense['date']       = expense['date'].strftime('%Y-%m-%d')
            expense['created_at'] = expense['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        cursor.close()
        connection.close()
        return jsonify(expenses)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/delete_expense/<int:expense_id>', methods=['DELETE'])
@login_required
def delete_expense(expense_id):
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        cursor = connection.cursor()
        # Only delete if it belongs to the current user
        cursor.execute("DELETE FROM expenses WHERE id = %s AND user_id = %s",
                       (expense_id, current_user.id))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'success': True, 'message': 'Expense deleted!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get_weekly_chart')
@login_required
def get_weekly_chart():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        cursor = connection.cursor()
        end_date   = datetime.now().date()
        start_date = end_date - timedelta(days=6)
        cursor.execute("""
            SELECT DATE(date) as expense_date, SUM(amount) as daily_total
            FROM expenses
            WHERE date BETWEEN %s AND %s AND user_id = %s
            GROUP BY DATE(date)
            ORDER BY expense_date
        """, (start_date, end_date, current_user.id))
        results = cursor.fetchall()
        cursor.close()

        week_data = {}
        current = start_date
        for _ in range(7):
            week_data[current.strftime('%Y-%m-%d')] = 0
            current += timedelta(days=1)
        for row in results:
            week_data[row[0].strftime('%Y-%m-%d')] = float(row[1])

        connection.close()
        return jsonify({'image': make_weekly_chart(week_data)})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get_monthly_chart')
@login_required
def get_monthly_chart():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'})
        cursor = connection.cursor()
        current_month_start = datetime.now().replace(day=1).date()
        cursor.execute("""
            SELECT category, SUM(amount) as category_total
            FROM expenses
            WHERE date >= %s AND user_id = %s
            GROUP BY category
            ORDER BY category_total DESC
        """, (current_month_start, current_user.id))
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        categories = [row[0] for row in results]
        amounts    = [float(row[1]) for row in results]
        return jsonify({'image': make_monthly_chart(categories, amounts)})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get_prediction')
@login_required
def get_prediction():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'prediction': None, 'message': 'Database connection failed'})
        prediction = predict_next_week(connection, current_user.id)
        connection.close()
        if prediction is None:
            return jsonify({'prediction': None,
                            'message': 'Need at least 5 days of expense data to make a prediction.'})
        return jsonify({'prediction': prediction,
                        'message': f'Predicted spending next week: ₹{prediction:.2f}'})
    except Exception as e:
        return jsonify({'prediction': None, 'message': str(e)})


@app.route('/get_stats')
@login_required
def get_stats():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({})
        cursor = connection.cursor()
        now         = datetime.now()
        month_start = now.replace(day=1).date()

        cursor.execute("SELECT COALESCE(SUM(amount),0) FROM expenses WHERE date >= %s AND user_id = %s",
                       (month_start, current_user.id))
        total_month = float(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM expenses WHERE date >= %s AND user_id = %s",
                       (month_start, current_user.id))
        count_month = cursor.fetchone()[0]

        cursor.execute("""
            SELECT category, SUM(amount) as total
            FROM expenses WHERE date >= %s AND user_id = %s
            GROUP BY category ORDER BY total DESC LIMIT 1
        """, (month_start, current_user.id))
        top_row      = cursor.fetchone()
        top_category = top_row[0] if top_row else '-'
        avg_daily    = total_month / now.day if now.day > 0 else 0

        cursor.close()
        connection.close()
        return jsonify({
            'total_month':  round(total_month, 2),
            'count_month':  count_month,
            'top_category': top_category,
            'avg_daily':    round(avg_daily, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    create_tables_if_not_exist()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
