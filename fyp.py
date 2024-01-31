import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import tkinter as tk
from tkinter import messagebox
import sqlite3
import hashlib

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Function to create a SQLite connection
def create_connection():
    conn = sqlite3.connect("user_credentials.db")
    return conn
# Function to create the users table if it doesn't exist
def create_users_table(conn):
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL)
    """
    conn.execute(query)
    conn.commit()
    # Function to insert user credentials into the database
def insert_user(conn, username, password):
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        query = "INSERT INTO users (username, password) VALUES (?, ?)"
        with conn:
            conn.execute(query, (username, password_hash))
        st.success(f"Signup successful for Username: {username}")
    except sqlite3.Error as e:
        st.error(f"Error occurred: {e}")
    except sqlite3.Error as e:
        st.error(f"Error occurred: {e}")
        conn.rollback()
    # Function to check user credentials during login
def authenticate_user(conn, username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    query = "SELECT * FROM users WHERE username = ? AND password = ?"
    cursor = conn.execute(query, (username, password_hash))
    user = cursor.fetchone()
    return user is not None

# Function for Signup
def signup():
    st.title("Signup Form")
    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", type="password")

    if st.button("Signup"):
        conn = create_connection()
        create_users_table(conn)
        insert_user(conn, username, password)
        st.success(f"Signup successful for Username: {username}")
        st.session_state.page = "Login"

# Function for Login
def login():
    st.title("Login Form")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        conn = create_connection()
        if authenticate_user(conn, username, password):
            st.success(f"Logged in as {username}")
            # if "page" not in st.session_state:
            st.session_state.page = "Fraud Detection"
        else:
            st.error("Invalid username or password")

# Function for Fraud Detection
def fraud_detection():
    st.title("Counterfeit Fraud Detection Model")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    # Create input fields for user to enter feature values
    input_df = st.text_input('Input All features')
    input_df_lst = input_df.split(',')
    # Create a button to submit input and get prediction
    submit = st.button("Submit")

    if submit:
        # Get input feature values
        try:
            import numpy as np  # Imported here for the np usage in this snippet
            features = np.array(input_df_lst, dtype=np.float64)
            # Make prediction (replace model with your trained model)
            prediction = model.predict(features.reshape(1,-1))
            # Display result
            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
        except ValueError:
            st.write("Invalid input format. Please provide comma-separated numerical values.")

def main():
    # st.sidebar.title("Authentication")
    # choice = st.sidebar.radio("Select an action", ["Signup", "Login"])
    if "page" not in st.session_state:
        st.session_state.page = "Signup"

    if st.session_state.page == "Signup":
        signup()
    elif st.session_state.page == "Login":
        login()
    elif st.session_state.page == "Fraud Detection":
        fraud_detection()

if __name__ == "__main__":
    main()
