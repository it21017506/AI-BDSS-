import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId

# 🔹 MongoDB Atlas connection
MONGO_URI = "mongodb+srv://predictionpro25_db_user:svVSBpZJSbSmtPKF@sales.q38zqvg.mongodb.net/?retryWrites=true&w=majority&appName=sales"

client = MongoClient(MONGO_URI)
db = client["sales_ai"]  # Database name
users_collection = db["users"]  # Users collection

# Streamlit page config
st.set_page_config(
    page_title="Login | Advanced Sales Prediction AI",
    page_icon="🔑",
    layout="centered"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .auth-box {
        background: rgba(102, 126, 234, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Init session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

def signup_user(email, password):
    if users_collection.find_one({"email": email}):
        st.error("⚠️ Email already exists. Try logging in.")
        return False
    users_collection.insert_one({"email": email, "password": password})
    st.success("✅ Signup successful! Please login now.")
    return True

def login_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and user["password"] == password:
        st.session_state.authenticated = True
        st.session_state.user_email = email
        st.success("✅ Login successful! Redirecting...")
        st.switch_page("pages/app.py")  # redirect to your main app
    else:
        st.error("❌ Invalid email or password")

# Auth UI
st.markdown('<h1 class="main-header">🔑 User Authentication </h1>', unsafe_allow_html=True)
mode = st.radio("Select option:", ["Login", "Sign Up"])

with st.container():
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)

    email = st.text_input("📧 Email")
    password = st.text_input("🔒 Password", type="password")

    if mode == "Sign Up":
        if st.button("📝 Sign Up"):
            if email and password:
                signup_user(email, password)
            else:
                st.warning("Please enter email and password.")
    else:
        if st.button("➡️ Login"):
            if email and password:
                login_user(email, password)
            else:
                st.warning("Please enter email and password.")

    st.markdown('</div>', unsafe_allow_html=True)
