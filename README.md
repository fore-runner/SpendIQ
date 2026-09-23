# SpendIQ – Smart Personal Expense Tracker

A full-stack personal finance web app built with Python, Flask, and MySQL. It helps track daily spending, visualize expenses with interactive charts, and estimate upcoming weekly expenses using machine learning.

**Try it out here:** [https://spendiq-production-3146.up.railway.app](https://spendiq-production-3146.up.railway.app)

---

## Features

- **User Accounts & Authentication:** Secure signup and login with hashed passwords via Flask-Login and Werkzeug. Each user has their own private dashboard and data.
- **Expense Logging:** Quick entry for amount, category, date, and description.
- **Visual Analytics:**
  - 7-day spending trend line chart powered by Seaborn & Matplotlib.
  - Monthly category breakdown bar chart.
- **ML Spending Forecast:** Linear regression model (scikit-learn) trained on your past 30 days of expense history to predict expected spending for the next week.
- **Filter & Search:** Filter transaction history by category or specific month.
- **Data Export:** One-click CSV export of all recorded expenses.

---

## Tech Stack

- **Backend:** Python 3, Flask, Flask-Login, Werkzeug
- **Database:** MySQL
- **Data Science & ML:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API, responsive layout)
- **Deployment:** Railway
