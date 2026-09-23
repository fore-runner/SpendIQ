import os
os.environ['MPLBACKEND'] = 'Agg'

from flask import Flask, render_template, request, jsonify
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# ── Database config ──────────────────────────────────────────
# Reads from environment variables (required for Railway deployment).
# For local dev, set a .env file or just hardcode below temporarily.
DB_CONFIG = {
    'host':     os.environ.get('MYSQLHOST',     'localhost'),
    'port':     int(os.environ.get('MYSQLPORT', 3306)),
    'database': os.environ.get('MYSQLDATABASE', 'expense_tracker'),
    'user':     os.environ.get('MYSQLUSER',     'root'),
    'password': os.environ.get('MYSQLPASSWORD', 'ayush'),  # set your local password here
}

# ── Database helpers ─────────────────────────────────────────

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None


def create_table_if_not_exists():
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    amount DECIMAL(10, 2) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    description VARCHAR(255),
                    date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

def predict_next_week(connection):
    try:
        cursor = connection.cursor()
        end_date   = datetime.now().date()
        start_date = end_date - timedelta(days=29)
        cursor.execute("""
            SELECT DATE(date) as d, SUM(amount) as total
            FROM expenses
            WHERE date BETWEEN %s AND %s
            GROUP BY DATE(date)
            ORDER BY d
        """, (start_date, end_date))
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


# ── Flask routes ──────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/add_expense', methods=['POST'])
def add_expense():
    try:
        data = request.get_json()
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO expenses (amount, category, description, date) VALUES (%s, %s, %s, %s)",
            (data['amount'], data['category'], data['description'], data['date'])
        )
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'success': True, 'message': 'Expense added successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get_expenses')
def get_expenses():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify([])
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM expenses ORDER BY date DESC")
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
def delete_expense(expense_id):
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        cursor = connection.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = %s", (expense_id,))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'success': True, 'message': 'Expense deleted!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get_weekly_chart')
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
            WHERE date BETWEEN %s AND %s
            GROUP BY DATE(date)
            ORDER BY expense_date
        """, (start_date, end_date))
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
            WHERE date >= %s
            GROUP BY category
            ORDER BY category_total DESC
        """, (current_month_start,))
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        categories = [row[0] for row in results]
        amounts    = [float(row[1]) for row in results]
        return jsonify({'image': make_monthly_chart(categories, amounts)})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get_prediction')
def get_prediction():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'prediction': None, 'message': 'Database connection failed'})
        prediction = predict_next_week(connection)
        connection.close()
        if prediction is None:
            return jsonify({'prediction': None,
                            'message': 'Need at least 5 days of expense data to make a prediction.'})
        return jsonify({'prediction': prediction,
                        'message': f'Predicted spending next week: ₹{prediction:.2f}'})
    except Exception as e:
        return jsonify({'prediction': None, 'message': str(e)})


@app.route('/get_stats')
def get_stats():
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({})
        cursor = connection.cursor()
        now         = datetime.now()
        month_start = now.replace(day=1).date()

        cursor.execute("SELECT COALESCE(SUM(amount),0) FROM expenses WHERE date >= %s", (month_start,))
        total_month = float(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM expenses WHERE date >= %s", (month_start,))
        count_month = cursor.fetchone()[0]

        cursor.execute("""
            SELECT category, SUM(amount) as total
            FROM expenses WHERE date >= %s
            GROUP BY category ORDER BY total DESC LIMIT 1
        """, (month_start,))
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
    create_table_if_not_exists()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
