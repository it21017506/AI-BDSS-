import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# Initialize session state for file format display
if 'show_file_formats' not in st.session_state:
    st.session_state.show_file_formats = True

# Page configuration
st.set_page_config(
    page_title="Advanced Sales Prediction AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
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
    .upload-section {
        background: rgba(102, 126, 234, 0.1);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 2rem 0;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .reinforcement-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'reinforcement_data' not in st.session_state:
    st.session_state.reinforcement_data = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None


@st.cache_data
def load_and_process_data(file):
    """Load and process the uploaded data with enhanced column detection"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Enhanced column detection for the new CSV format
        date_col = None
        product_col = None
        sales_per_unit_col = None
        quantity_col = None
        total_sales_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'order' in col_lower:
                date_col = col
            elif 'product' in col_lower or 'category' in col_lower:
                product_col = col
            elif 'sales per unit' in col_lower or 'price' in col_lower:
                sales_per_unit_col = col
            elif 'quantity' in col_lower or 'qty' in col_lower:
                quantity_col = col
            elif 'total sales' in col_lower or 'total' in col_lower:
                total_sales_col = col

        # Fallback column detection
        if date_col is None:
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break

        if product_col is None:
            for col in df.columns:
                if 'product' in col.lower() or 'category' in col.lower():
                    product_col = col
                    break

        if total_sales_col is None:
            for col in df.columns:
                if 'total' in col.lower():
                    total_sales_col = col
                    break

        if quantity_col is None:
            for col in df.columns:
                if 'quantity' in col.lower():
                    quantity_col = col
                    break

        if sales_per_unit_col is None:
            for col in df.columns:
                if 'sales per unit' in col.lower():
                    sales_per_unit_col = col
                    break

        # Validate required columns
        required_cols = [date_col, product_col, total_sales_col]
        if any(col is None for col in required_cols):
            st.error(
                "Could not detect required columns. Please ensure your file has: Date, Product Category, and Total Sales columns.")
            st.error(f"Available columns: {list(df.columns)}")
            return None

        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        # Convert numeric columns
        if total_sales_col:
            df[total_sales_col] = pd.to_numeric(df[total_sales_col], errors='coerce')
        if quantity_col:
            df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')
        if sales_per_unit_col:
            df[sales_per_unit_col] = pd.to_numeric(df[sales_per_unit_col], errors='coerce')

        # Rename columns for consistency
        column_mapping = {
            date_col: 'date',
            product_col: 'product',
            total_sales_col: 'total_sales'
        }

        if quantity_col:
            column_mapping[quantity_col] = 'quantity'
        if sales_per_unit_col:
            column_mapping[sales_per_unit_col] = 'sales_per_unit'

        df = df.rename(columns=column_mapping)

        # Sort by date
        df = df.sort_values('date')

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def create_advanced_regression_models():
    """Create LSTM, Lasso, and Ridge regression models for ensemble prediction"""
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1)
    }
    return models


def create_quick_prediction_models():
    """Create quick prediction models using Lasso, Ridge, XGBoost, and AdaBoost"""
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42)
    }
    return models


def create_lstm_model(input_shape):
    """Create optimized LSTM model for 6.5GB RAM with efficient training"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),  # Reduced from 128
        Dropout(0.2),  # Reduced from 0.3
        LSTM(16, return_sequences=False),  # Reduced from 64, removed one layer
        Dropout(0.2),  # Reduced from 0.3
        Dense(8, activation='relu'),  # Reduced from 16
        Dense(1)
    ])

    # Use a more memory-efficient optimizer
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
    return model


def prepare_features_for_regression(df, product, target='total_sales'):
    """Prepare features for regression models with memory optimization"""
    product_data = df[df['product'] == product].copy()

    if len(product_data) < 10:
        return None, None, None

    # Limit data size for memory efficiency in deep learning
    if len(product_data) > 500:  # Limit to 500 samples for LSTM training
        product_data = product_data.sample(n=500, random_state=42)

    # Create time-based features
    product_data['year'] = product_data['date'].dt.year
    product_data['month'] = product_data['date'].dt.month
    product_data['day'] = product_data['date'].dt.day
    product_data['day_of_week'] = product_data['date'].dt.dayofweek
    product_data['quarter'] = product_data['date'].dt.quarter

    # Create lag features for the target variable
    product_data = product_data.sort_values('date')
    product_data[f'{target}_lag_1'] = product_data[target].shift(1)
    product_data[f'{target}_lag_2'] = product_data[target].shift(2)
    product_data[f'{target}_lag_3'] = product_data[target].shift(3)

    # Create rolling features for the target variable
    product_data[f'{target}_rolling_mean_3'] = product_data[target].rolling(window=3).mean()
    product_data[f'{target}_rolling_mean_7'] = product_data[target].rolling(window=7).mean()
    product_data[f'{target}_rolling_std_3'] = product_data[target].rolling(window=3).std()

    # Add cross-features if both quantity and total_sales are available
    if 'quantity' in product_data.columns and 'total_sales' in product_data.columns:
        if target == 'total_sales':
            product_data['quantity_lag_1'] = product_data['quantity'].shift(1)
            product_data['quantity_rolling_mean_3'] = product_data['quantity'].rolling(window=3).mean()
        elif target == 'quantity':
            product_data['total_sales_lag_1'] = product_data['total_sales'].shift(1)
            product_data['total_sales_rolling_mean_3'] = product_data['total_sales'].rolling(window=3).mean()

    # Drop NaN values
    product_data = product_data.dropna()

    if len(product_data) < 5:
        return None, None, None

    # Prepare features
    feature_columns = ['year', 'month', 'day', 'day_of_week', 'quarter',
                       f'{target}_lag_1', f'{target}_lag_2', f'{target}_lag_3',
                       f'{target}_rolling_mean_3', f'{target}_rolling_mean_7', f'{target}_rolling_std_3']

    # Add cross-features
    if target == 'total_sales' and 'quantity' in product_data.columns:
        feature_columns.extend(['quantity_lag_1', 'quantity_rolling_mean_3'])
    elif target == 'quantity' and 'total_sales' in product_data.columns:
        feature_columns.extend(['total_sales_lag_1', 'total_sales_rolling_mean_3'])

    # Add sales_per_unit if available
    if 'sales_per_unit' in product_data.columns:
        feature_columns.append('sales_per_unit')

    X = product_data[feature_columns]
    y = product_data[target]

    return X, y, product_data


def train_quick_models(X, y, product, target_type):
    """Train quick prediction models using Lasso, Ridge, XGBoost, and AdaBoost"""
    models = create_quick_prediction_models()
    trained_models = {}
    predictions = {}
    scores = {}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train all models
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            trained_models[name] = model
            predictions[name] = y_pred
            scores[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'RMSE': np.sqrt(mse)
            }

        except Exception as e:
            st.warning(f"Error training {name} for {product} ({target_type}): {str(e)}")
            continue

    return trained_models, predictions, scores, scaler


def train_ensemble_models(X, y, product, target_type):
    """Train LSTM, Lasso, and Ridge regression models for both quantity and total sales"""
    models = create_advanced_regression_models()
    trained_models = {}
    predictions = {}
    scores = {}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train LSTM model with memory optimization
    try:
        # Limit data size for memory efficiency
        max_samples = min(1000, len(X_train_scaled))  # Limit to 1000 samples max
        if len(X_train_scaled) > max_samples:
            # Take a random sample for training
            indices = np.random.choice(len(X_train_scaled), max_samples, replace=False)
            X_train_scaled = X_train_scaled[indices]
            y_train = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]

        # Prepare LSTM data - ensure proper shape
        if len(X_train_scaled.shape) == 2:
            X_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        else:
            # If already 3D, use as is
            X_lstm = X_train_scaled
            X_test_lstm = X_test_scaled

        # Create and train LSTM model
        lstm_model = create_lstm_model((X_lstm.shape[1], 1))

        # Memory-efficient early stopping with reduced patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced from 10
            restore_best_weights=True
        )

        # Memory-efficient training with smaller batch size and fewer epochs
        lstm_history = lstm_model.fit(
            X_lstm, y_train,
            epochs=20,  # Reduced from 50
            batch_size=16,  # Reduced from 32
            validation_data=(X_test_lstm, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Make LSTM predictions with memory optimization
        y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0, batch_size=16).flatten()

        # Calculate LSTM metrics
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
        r2_lstm = r2_score(y_test, y_pred_lstm)

        trained_models['LSTM'] = lstm_model
        predictions['LSTM'] = y_pred_lstm
        scores['LSTM'] = {
            'MSE': mse_lstm,
            'MAE': mae_lstm,
            'R2': r2_lstm,
            'RMSE': np.sqrt(mse_lstm)
        }

        # Clear memory
        del X_lstm, X_test_lstm, lstm_history
        tf.keras.backend.clear_session()

    except Exception as e:
        st.warning(f"Error training LSTM for {product} ({target_type}): {str(e)}")
        st.info(f"LSTM training failed for {product}. Continuing with regression models only.")
        # Clear memory on error
        tf.keras.backend.clear_session()

    # Train regression models
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            trained_models[name] = model
            predictions[name] = y_pred
            scores[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'RMSE': np.sqrt(mse)
            }

        except Exception as e:
            st.warning(f"Error training {name} for {product} ({target_type}): {str(e)}")
            continue

    return trained_models, predictions, scores, scaler


def predict_future_regression(trained_models, scaler, last_data, steps=6, target_type='total_sales'):
    """Predict future values using ensemble models with memory optimization"""
    predictions = {}

    for name, model in trained_models.items():
        try:
            future_predictions = []
            current_data = last_data.copy()

            for _ in range(steps):
                # Prepare features for next prediction
                features = current_data.reshape(1, -1)
                features_scaled = scaler.transform(features)

                # Handle LSTM vs regression models differently
                if name == 'LSTM':
                    # Reshape for LSTM input with memory optimization
                    features_lstm = features_scaled.reshape(1, features_scaled.shape[1], 1)
                    next_pred = model.predict(features_lstm, verbose=0, batch_size=1)[0, 0]
                else:
                    # Regular regression model
                    next_pred = model.predict(features_scaled)[0]

                future_predictions.append(next_pred)

                # Update features for next iteration (simplified)
                # In a real scenario, you'd need to update all time-based features
                current_data[0] = next_pred  # Update the target variable

            predictions[name] = future_predictions

        except Exception as e:
            st.warning(f"Error predicting with {name} ({target_type}): {str(e)}")
            continue

    return predictions


def create_reinforcement_learning_agent():
    """Create a simple reinforcement learning agent for business optimization"""

    class BusinessRLAgent:
        def __init__(self, n_actions=5):
            self.n_actions = n_actions
            self.q_table = {}
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.epsilon = 0.1

        def get_state_key(self, state):
            return str(state)

        def get_action(self, state):
            state_key = self.get_state_key(state)

            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)

            # Epsilon-greedy strategy
            if np.random.random() < self.epsilon:
                return np.random.randint(self.n_actions)
            else:
                return np.argmax(self.q_table[state_key])

        def update(self, state, action, reward, next_state):
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)

            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.n_actions)

            # Q-learning update
            current_q = self.q_table[state_key][action]
            max_next_q = np.max(self.q_table[next_state_key])
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[state_key][action] = new_q

    return BusinessRLAgent()


def generate_enhanced_business_recommendations(df, feedback_data, product_models, predictions):
    """Generate enhanced business recommendations using reinforcement learning and feedback analysis"""
    recommendations = {}

    for product in df['product'].unique():
        # Generate recommendations for all products, not just those with trained models
        product_data = df[df['product'] == product]

        # Calculate business metrics
        avg_sales = product_data['total_sales'].mean()
        sales_volatility = product_data['total_sales'].std()
        total_revenue = product_data['total_sales'].sum()

        # Analyze feedback data if available
        feedback_analysis = {}
        if feedback_data is not None:
            product_feedback = feedback_data[feedback_data['Product Category'] == product]
            if len(product_feedback) > 0:
                feedback_analysis = {
                    'avg_rating': product_feedback['Rating'].mean(),
                    'total_reviews': len(product_feedback),
                    'positive_reviews': len(product_feedback[product_feedback['Rating'] >= 4]),
                    'negative_reviews': len(product_feedback[product_feedback['Rating'] <= 2]),
                    'review_sentiment': analyze_review_sentiment(product_feedback['Review'].tolist())
                }

        # Create RL agent
        agent = create_reinforcement_learning_agent()

        # Simulate business scenarios with feedback
        scenarios = []
        for _ in range(100):
            # Define state based on current metrics and feedback
            state = [
                int(avg_sales > product_data['total_sales'].median()),
                int(sales_volatility > product_data['total_sales'].std()),
                int(total_revenue > product_data['total_sales'].sum() * 0.8)
            ]

            # Add feedback-based state if available
            if feedback_analysis:
                state.extend([
                    int(feedback_analysis.get('avg_rating', 0) > 3.5),
                    int(feedback_analysis.get('positive_reviews', 0) > feedback_analysis.get('negative_reviews', 0))
                ])

            action = agent.get_action(state)

            # Calculate reward based on predicted performance and feedback
            predicted_avg = avg_sales  # Default to current average
            if product in predictions:
                # Check if we have ensemble predictions
                if 'ensemble' in predictions[product]:
                    predicted_avg = np.mean(predictions[product]['ensemble'])
                elif 'total_sales' in predictions[product] and 'ensemble' in predictions[product]['total_sales']:
                    predicted_avg = np.mean(predictions[product]['total_sales']['ensemble'])
                elif 'quantity' in predictions[product] and 'ensemble' in predictions[product]['quantity']:
                    predicted_avg = np.mean(predictions[product]['quantity']['ensemble'])

            reward = 1 if predicted_avg > avg_sales else -0.5

            # Adjust reward based on feedback
            if feedback_analysis:
                feedback_bonus = (feedback_analysis.get('avg_rating', 3) - 3) * 0.2
                reward += feedback_bonus

            # Update agent
            next_state = state.copy()
            agent.update(state, action, reward, next_state)

            scenarios.append({
                'action': action,
                'reward': reward,
                'state': state
            })

        # Generate enhanced recommendations
        best_action = np.argmax([s['reward'] for s in scenarios])

        action_descriptions = {
            0: "Increase marketing budget and improve brand awareness",
            1: "Optimize pricing strategy based on customer feedback",
            2: "Improve inventory management and supply chain",
            3: "Enhance customer service and support",
            4: "Expand product line with customer-driven features"
        }

        # Generate detailed suggestions based on feedback
        detailed_suggestions = generate_detailed_suggestions(product, feedback_analysis, product_data)

        recommendations[product] = {
            'recommended_action': action_descriptions[best_action],
            'confidence_score': np.mean([s['reward'] for s in scenarios]),
            'current_avg_sales': avg_sales,
            'predicted_improvement': predicted_avg - avg_sales,
            'risk_level': 'High' if sales_volatility > avg_sales * 0.5 else 'Low',
            'feedback_analysis': feedback_analysis,
            'detailed_suggestions': detailed_suggestions
        }

    return recommendations


def analyze_review_sentiment(reviews):
    """Analyze sentiment of reviews using MobileBERT and regex patterns"""
    if not reviews:
        return {}

    # Load Sentiment Pipeline with MobileBERT (Load once globally)
    sentiment_pipeline = pipeline('sentiment-analysis', model='google/mobilebert-uncased')

    # Precompile Keyword Patterns
    delivery_pattern = re.compile(
        r'\b(delivery|delivered|late|delay|shipping|arrival|tracking|courier|logistics|ETA|out for delivery|reschedule)\b',
        re.IGNORECASE)
    quality_pattern = re.compile(r'\b(quality|poor|not good|detailing|cheap|bad|worth|flawless|superior)\b',
                                 re.IGNORECASE)

    sentiment_analysis = {
        'positive_count': 0,
        'negative_count': 0,
        'delivery_issues': 0,
        'quality_issues': 0
    }

    # Run Sentiment Analysis in Batches
    results = sentiment_pipeline(reviews, batch_size=32)  # Larger batch = faster on CPU

    # Vectorized Keyword Matching
    import pandas as pd
    feedback_df = pd.DataFrame({'Review': reviews})
    delivery_flags = feedback_df['Review'].str.contains(delivery_pattern, regex=True)
    quality_flags = feedback_df['Review'].str.contains(quality_pattern, regex=True)

    # Count Results
    for idx, result in enumerate(results):
        label = result['label']
        if label.upper() == 'POSITIVE':
            sentiment_analysis['positive_count'] += 1
        elif label.upper() == 'NEGATIVE':
            sentiment_analysis['negative_count'] += 1

        if delivery_flags.iloc[idx]:
            sentiment_analysis['delivery_issues'] += 1
        if quality_flags.iloc[idx]:
            sentiment_analysis['quality_issues'] += 1

    return sentiment_analysis


def generate_detailed_suggestions(product, feedback_analysis, product_data):
    """Generate detailed business suggestions based on feedback and sales data"""
    suggestions = []

    if not feedback_analysis:
        return ["No feedback data available for detailed analysis"]

    avg_rating = feedback_analysis.get('avg_rating', 0)
    total_reviews = feedback_analysis.get('total_reviews', 0)
    sentiment = feedback_analysis.get('review_sentiment', {})

    # Rating-based suggestions
    if avg_rating < 2.5:
        suggestions.append(
            "🚨 **Critical Issue**: Low customer satisfaction. Focus on quality improvement and customer service training.")
    elif avg_rating < 3.5:
        suggestions.append(
            "⚠️ **Improvement Needed**: Moderate customer satisfaction. Implement feedback-based improvements.")
    else:
        suggestions.append(
            "✅ **Good Performance**: High customer satisfaction. Maintain quality standards and consider expansion.")

    # Review volume suggestions
    if total_reviews < 5:
        suggestions.append(
            "📊 **Limited Feedback**: Encourage more customer reviews to better understand product performance.")
    elif total_reviews > 20:
        suggestions.append("📈 **Strong Feedback Base**: Use customer insights for product development and marketing.")

    # Sentiment-based suggestions
    if sentiment.get('delivery_issues', 0) > 0:
        suggestions.append(
            "🚚 **Delivery Issues**: Improve delivery speed and handling procedures. Consider logistics optimization.")

    if sentiment.get('quality_issues', 0) > 0:
        suggestions.append("🔧 **Quality Concerns**: Enhance product quality control and manufacturing processes.")

    # Sales-based suggestions
    avg_sales = product_data['total_sales'].mean()
    if avg_sales < 100:
        suggestions.append("💰 **Low Sales Volume**: Consider promotional campaigns and market expansion strategies.")
    elif avg_sales > 500:
        suggestions.append(
            "💎 **High Performance**: Focus on maintaining market position and premium pricing strategies.")

    # Cross-selling opportunities
    if 'quantity' in product_data.columns:
        avg_quantity = product_data['quantity'].mean()
        if avg_quantity < 2:
            suggestions.append("🛒 **Bundle Opportunities**: Create product bundles to increase average order value.")

    return suggestions


def main_page():
    """Main dashboard page"""
    # Add logout button in top right
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown('<h1 class="main-header">🚀 Advanced Sales Prediction AI</h1>', unsafe_allow_html=True)
        st.markdown(
            "### Transform your transaction data into actionable sales insights with Advanced AI & Reinforcement Learning")

    with col3:
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.user_email = None
            st.rerun()

    # Display user info
    if st.session_state.user_email:
        st.info(f"👤 Welcome, {st.session_state.user_email}")

    # File format requirements section
    if st.session_state.show_file_formats:
        with st.expander("📋 Required File Formats", expanded=True):
            st.markdown("""
            ### Sales Data (sample_data_new_correct.csv):
            ```
            Order Date,Product Category,Sales per Unit,Quantity,Total Sales
            7/21/2024,Clothing,100,1,100
            7/20/2024,Clothing,100,1,100
            ```

            ### Feedback Data (feedback.csv):
            ```
            Product Category,Rating,Review
            Clothing,4,The delivery team handled the product with care.
            Clothing,3,Had slight delays but the product was in good shape.
            ```
            """)
            if st.button("Got it!", key="hide_formats"):
                st.session_state.show_file_formats = False
                st.rerun()

    # Navigation buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Historical Sales Prediction", type="primary", use_container_width=True):
            st.session_state.current_page = 'historical'
            st.rerun()

    with col2:
        if st.button("🤖 Reinforcement Learning", type="primary", use_container_width=True):
            st.session_state.current_page = 'reinforcement'
            st.rerun()

    with col3:
        if st.button("📈 Business Analytics", type="primary", use_container_width=True):
            st.session_state.current_page = 'analytics'
            st.rerun()

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your Transaction Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your transaction data with columns: Order Date, Product Category, Sales per Unit, Quantity, Total Sales"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("🔄 Loading and processing your data..."):
            df = load_and_process_data(uploaded_file)
            st.session_state.uploaded_data = df

        if df is not None:
            st.success(
                f"✅ Data loaded successfully! Found {len(df)} transactions across {df['product'].nunique()} products")

            # Store data in session state
            st.session_state.uploaded_data = df

            # Data overview
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Transactions", len(df))

            with col2:
                st.metric("Unique Products", df['product'].nunique())

            with col3:
                st.metric("Date Range",
                          f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

            with col4:
                st.metric("Total Revenue", f"${df['total_sales'].sum():,.0f}")


def historical_prediction_page():
    """Historical sales prediction page with separate quantity and total sales models"""
    st.markdown('<h1 class="main-header">📊 Historical Sales Prediction</h1>', unsafe_allow_html=True)

    # Back button
    if st.button("← Back to Main Menu"):
        st.session_state.current_page = 'main'
        st.rerun()

    if st.session_state.uploaded_data is None:
        st.warning("Please upload data from the main menu first.")
        return

    df = st.session_state.uploaded_data

    # Display all product categories
    st.markdown("## 📋 Available Product Categories")
    all_products = df['product'].unique()
    st.info(f"Found {len(all_products)} product categories: {', '.join(all_products)}")

    # Product selection
    st.markdown("## 🎯 Select Products for Advanced Analysis")
    selected_products = st.multiselect(
        "Choose products to analyze",
        all_products,
        default=all_products[:min(5, len(all_products))],
        help="Select products for advanced regression analysis and prediction"
    )

    if selected_products:
        # Prediction settings
        col1, col2, col3 = st.columns(3)
        with col1:
            prediction_period = st.selectbox(
                "Select Prediction Period",
                ["1 Month", "3 Months", "6 Months", "1 Year"],
                index=2
            )

        with col2:
            target_variable = st.selectbox(
                "Select Target Variable",
                ["Both", "Quantity Only", "Total Sales Only"],
                index=0,
                help="Choose which variables to predict"
            )

        with col3:
            show_ensemble = st.checkbox(
                "Show Ensemble Model Comparison",
                value=True
            )

        # Training options
        st.markdown("### 🎯 Training Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⚡ Quick Prediction (Fast)", type="primary", use_container_width=True):
                with st.spinner("Training quick models..."):
                    train_quick_prediction_models(selected_products, target_variable, show_ensemble)

        with col2:
            if st.button("🧠 Deep Learning (Advanced)", type="primary", use_container_width=True):
                with st.spinner("Training advanced models..."):
                    train_advanced_prediction_models(selected_products, target_variable, show_ensemble)

        with col3:
            if st.button("🔮 Hybrid Model (XGBoost + LSTM + SGD)", type="primary", use_container_width=True):
                with st.spinner("Training hybrid models..."):
                    train_hybrid_prediction_models(selected_products, target_variable, show_ensemble)


def display_prediction_results(predictions, scores, show_ensemble, target_variable="Both"):
    """Display prediction results with advanced visualizations for both quantity and total sales"""
    if not predictions:
        st.warning("No predictions available. Please check your data and try again.")
        return

    st.markdown("## 📈 Advanced Prediction Results")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "📊 Model Performance", "📈 Ensemble Analysis"])

    with tab1:
        st.markdown("### Future Predictions")

        # Display all products at once
        for selected_product in predictions.keys():
            st.markdown(f"#### 📊 Predictions for {selected_product}")

            # Check if we have both quantity and total sales predictions
            has_quantity = 'quantity' in predictions[selected_product]
            has_total_sales = 'total_sales' in predictions[selected_product]

            if target_variable == "Both" and has_quantity and has_total_sales:
                # Create subplots for both quantity and total sales
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Quantity Predictions', 'Total Sales Predictions'),
                    vertical_spacing=0.1
                )

                # Quantity predictions
                historical_data_qty = st.session_state.product_models[selected_product]['quantity']['data']

                fig.add_trace(go.Scatter(
                    x=historical_data_qty['date'],
                    y=historical_data_qty['quantity'],
                    mode='lines+markers',
                    name=f"{selected_product} Quantity (Historical)",
                    line=dict(width=2)
                ), row=1, col=1)

                future_dates = pd.date_range(
                    start=historical_data_qty['date'].iloc[-1] + pd.DateOffset(days=1),
                    periods=len(predictions[selected_product]['quantity']['ensemble']),
                    freq='M'
                )

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions[selected_product]['quantity']['ensemble'],
                    mode='lines+markers',
                    name=f"{selected_product} Quantity (Predicted)",
                    line=dict(width=2, dash='dash'),
                    marker=dict(symbol='diamond')
                ), row=1, col=1)

                # Total sales predictions
                historical_data_sales = st.session_state.product_models[selected_product]['total_sales']['data']

                fig.add_trace(go.Scatter(
                    x=historical_data_sales['date'],
                    y=historical_data_sales['total_sales'],
                    mode='lines+markers',
                    name=f"{selected_product} Sales (Historical)",
                    line=dict(width=2)
                ), row=2, col=1)

                future_dates = pd.date_range(
                    start=historical_data_sales['date'].iloc[-1] + pd.DateOffset(days=1),
                    periods=len(predictions[selected_product]['total_sales']['ensemble']),
                    freq='M'
                )

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions[selected_product]['total_sales']['ensemble'],
                    mode='lines+markers',
                    name=f"{selected_product} Sales (Predicted)",
                    line=dict(width=2, dash='dash'),
                    marker=dict(symbol='diamond')
                ), row=2, col=1)

                fig.update_layout(height=1000, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            elif target_variable == "Both" and (has_quantity or has_total_sales):
                # Handle case where only one variable has predictions
                if has_quantity:
                    target_type = 'quantity'
                else:
                    target_type = 'total_sales'

                fig = go.Figure()
                historical_data = st.session_state.product_models[selected_product][target_type]['data']

                fig.add_trace(go.Scatter(
                    x=historical_data['date'],
                    y=historical_data[target_type],
                    mode='lines+markers',
                    name=f"{selected_product} {target_type.title()} (Historical)",
                    line=dict(width=2)
                ))

                future_dates = pd.date_range(
                    start=historical_data['date'].iloc[-1] + pd.DateOffset(days=1),
                    periods=len(predictions[selected_product][target_type]['ensemble']),
                    freq='M'
                )

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions[selected_product][target_type]['ensemble'],
                    mode='lines+markers',
                    name=f"{selected_product} {target_type.title()} (Predicted)",
                    line=dict(width=2, dash='dash'),
                    marker=dict(symbol='diamond')
                ))

                fig.update_layout(
                    title=f"Historical vs Predicted {target_type.title()} for {selected_product}",
                    xaxis_title="Date",
                    yaxis_title=f"{target_type.title()}",
                    height=1000,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                # Single variable prediction
                target_type = 'quantity' if target_variable == "Quantity Only" else 'total_sales'
                if target_type in predictions[selected_product]:
                    fig = go.Figure()

                    historical_data = st.session_state.product_models[selected_product][target_type]['data']

                    fig.add_trace(go.Scatter(
                        x=historical_data['date'],
                        y=historical_data[target_type],
                        mode='lines+markers',
                        name=f"{selected_product} {target_type.title()} (Historical)",
                        line=dict(width=2)
                    ))

                    future_dates = pd.date_range(
                        start=historical_data['date'].iloc[-1] + pd.DateOffset(days=1),
                        periods=len(predictions[selected_product][target_type]['ensemble']),
                        freq='M'
                    )

                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions[selected_product][target_type]['ensemble'],
                        mode='lines+markers',
                        name=f"{selected_product} {target_type.title()} (Predicted)",
                        line=dict(width=2, dash='dash'),
                        marker=dict(symbol='diamond')
                    ))

                    fig.update_layout(
                        title=f"Historical vs Predicted {target_type.title()} for {selected_product}",
                        xaxis_title="Date",
                        yaxis_title=f"{target_type.title()}",
                        height=1000,
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No predictions available for {selected_product} with target {target_type}")

            st.markdown("---")  # Add separator between products

    with tab2:
        st.markdown("### Model Performance Comparison")

        if show_ensemble and scores:
            # Create performance comparison table
            performance_data = []

            for product, product_scores in scores.items():
                for target_type, target_scores in product_scores.items():
                    for model_name, metrics in target_scores.items():
                        performance_data.append({
                            'Product': product,
                            'Target': target_type.replace('_', ' ').title(),
                            'Model': model_name,
                            'R² Score': f"{metrics['R2']:.3f}",
                            'RMSE': f"${metrics['RMSE']:.2f}",
                            'MAE': f"${metrics['MAE']:.2f}"
                        })

            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)

            # Performance heatmap for selected target
            if target_variable == "Both":
                target_selection = st.selectbox(
                    "Select Target Variable for Heatmap",
                    ["quantity", "total_sales"],
                    format_func=lambda x: x.replace('_', ' ').title()
                )

                # Filter data for selected target
                filtered_data = [row for row in performance_data if
                                 row['Target'].lower().replace(' ', '_') == target_selection]
                if filtered_data:
                    filtered_df = pd.DataFrame(filtered_data)
                    pivot_df = filtered_df.pivot(index='Product', columns='Model', values='R² Score')
                    pivot_df = pivot_df.apply(pd.to_numeric, errors='coerce')

                    fig = px.imshow(
                        pivot_df,
                        title=f"Model Performance Heatmap - {target_selection.replace('_', ' ').title()} (R² Scores)",
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Single target variable
                target_type = 'quantity' if target_variable == "Quantity Only" else 'total_sales'
                filtered_data = [row for row in performance_data if
                                 row['Target'].lower().replace(' ', '_') == target_type]
                if filtered_data:
                    filtered_df = pd.DataFrame(filtered_data)
                    pivot_df = filtered_df.pivot(index='Product', columns='Model', values='R² Score')
                    pivot_df = pivot_df.apply(pd.to_numeric, errors='coerce')

                    fig = px.imshow(
                        pivot_df,
                        title=f"Model Performance Heatmap - {target_type.replace('_', ' ').title()} (R² Scores)",
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Ensemble Model Analysis")

        if predictions:
            # Product selection for ensemble analysis
            if len(predictions) > 1:
                selected_product_ensemble = st.selectbox(
                    "Select Product for Ensemble Analysis",
                    list(predictions.keys()),
                    key="ensemble_product"
                )
            else:
                selected_product_ensemble = list(predictions.keys())[0]

            # Target selection for ensemble analysis
            available_targets = list(predictions[selected_product_ensemble].keys())
            if len(available_targets) > 1:
                selected_target = st.selectbox(
                    "Select Target Variable for Ensemble Analysis",
                    available_targets,
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="ensemble_target"
                )
            else:
                selected_target = available_targets[0]

            if selected_target in predictions[selected_product_ensemble]:
                pred_data = predictions[selected_product_ensemble][selected_target]

                # Compare individual models vs ensemble
                fig = go.Figure()

                # Plot individual model predictions
                for model_name, model_preds in pred_data['individual'].items():
                    if len(model_preds) > 0:
                        fig.add_trace(go.Scatter(
                            x=list(range(len(model_preds))),
                            y=model_preds,
                            mode='lines+markers',
                            name=f"{model_name}",
                            line=dict(width=1, dash='dot')
                        ))

                # Plot ensemble prediction
                if 'ensemble' in pred_data:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(pred_data['ensemble']))),
                        y=pred_data['ensemble'],
                        mode='lines+markers',
                        name=f"Ensemble",
                        line=dict(width=3),
                        marker=dict(symbol='diamond', size=8)
                    ))

                fig.update_layout(
                    title=f"Individual Models vs Ensemble Predictions - {selected_product_ensemble} ({selected_target.replace('_', ' ').title()})",
                    xaxis_title="Prediction Steps",
                    yaxis_title=f"Predicted {selected_target.replace('_', ' ').title()}",
                    height=1000,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)


def train_quick_prediction_models(selected_products, target_variable, show_ensemble):
    """Train quick prediction models"""
    df = st.session_state.uploaded_data
    product_models = {}
    all_predictions = {}
    all_scores = {}

    total_models = len(selected_products)
    if target_variable == "Both":
        total_models *= 2  # Two models per product

    current_model = 0

    for i, product in enumerate(selected_products):
        st.text(f"Training quick models for {product}...")

        # Train models based on target variable selection
        if target_variable in ["Both", "Quantity Only"] and 'quantity' in df.columns:
            # Train quantity model
            X_qty, y_qty, product_data_qty = prepare_features_for_regression(df, product, 'quantity')
            if X_qty is not None:
                trained_models_qty, predictions_qty, scores_qty, scaler_qty = train_quick_models(
                    X_qty, y_qty, product, 'quantity'
                )

                if trained_models_qty:
                    if product not in product_models:
                        product_models[product] = {}
                    product_models[product]['quantity'] = {
                        'models': trained_models_qty,
                        'scaler': scaler_qty,
                        'data': product_data_qty
                    }

                    # Predict future quantities
                    last_data_qty = X_qty.iloc[-1].values
                    future_predictions_qty = predict_future_regression(
                        trained_models_qty, scaler_qty, last_data_qty, steps=6, target_type='quantity'
                    )

                    # Ensure we have at least some predictions
                    if not future_predictions_qty:
                        st.warning(f"No predictions generated for {product} quantity. Skipping this product.")
                        continue

                    if future_predictions_qty:
                        ensemble_predictions_qty = []
                        for step in range(6):
                            step_predictions = [pred[step] for pred in future_predictions_qty.values() if
                                                len(pred) > step]
                            if step_predictions:
                                ensemble_predictions_qty.append(np.mean(step_predictions))

                        if product not in all_predictions:
                            all_predictions[product] = {}
                        all_predictions[product]['quantity'] = {
                            'individual': future_predictions_qty,
                            'ensemble': ensemble_predictions_qty
                        }

                        if product not in all_scores:
                            all_scores[product] = {}
                        all_scores[product]['quantity'] = scores_qty

                current_model += 1

        if target_variable in ["Both", "Total Sales Only"]:
            # Train total sales model
            X_sales, y_sales, product_data_sales = prepare_features_for_regression(df, product, 'total_sales')
            if X_sales is not None:
                trained_models_sales, predictions_sales, scores_sales, scaler_sales = train_quick_models(
                    X_sales, y_sales, product, 'total_sales'
                )

                if trained_models_sales:
                    if product not in product_models:
                        product_models[product] = {}
                    product_models[product]['total_sales'] = {
                        'models': trained_models_sales,
                        'scaler': scaler_sales,
                        'data': product_data_sales
                    }

                    # Predict future sales
                    last_data_sales = X_sales.iloc[-1].values
                    future_predictions_sales = predict_future_regression(
                        trained_models_sales, scaler_sales, last_data_sales, steps=6, target_type='total_sales'
                    )

                    # Ensure we have at least some predictions
                    if not future_predictions_sales:
                        st.warning(f"No predictions generated for {product} total sales. Skipping this product.")
                        continue

                    if future_predictions_sales:
                        ensemble_predictions_sales = []
                        for step in range(6):
                            step_predictions = [pred[step] for pred in future_predictions_sales.values() if
                                                len(pred) > step]
                            if step_predictions:
                                ensemble_predictions_sales.append(np.mean(step_predictions))

                        if product not in all_predictions:
                            all_predictions[product] = {}
                        all_predictions[product]['total_sales'] = {
                            'individual': future_predictions_sales,
                            'ensemble': ensemble_predictions_sales
                        }

                        if product not in all_scores:
                            all_scores[product] = {}
                        all_scores[product]['total_sales'] = scores_sales

                current_model += 1

    st.success("✅ Quick models trained successfully!")

    # Store results in session state
    st.session_state.product_models = product_models
    st.session_state.all_predictions = all_predictions
    st.session_state.all_scores = all_scores
    st.session_state.models_trained = True

    # Display results
    display_prediction_results(all_predictions, all_scores, show_ensemble, target_variable)


def train_advanced_prediction_models(selected_products, target_variable, show_ensemble):
    """Train advanced deep learning models with memory optimization for 6.5GB RAM"""
    df = st.session_state.uploaded_data
    product_models = {}
    all_predictions = {}
    all_scores = {}

    total_models = len(selected_products)
    if target_variable == "Both":
        total_models *= 2  # Two models per product

    current_model = 0

    # Progress bar for advanced training
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, product in enumerate(selected_products):
        status_text.text(f"🧠 Training advanced models for {product}... (Memory optimized)")
        progress_bar.progress((i + 1) / len(selected_products))

        # Train models based on target variable selection
        if target_variable in ["Both", "Quantity Only"] and 'quantity' in df.columns:
            # Train quantity model
            X_qty, y_qty, product_data_qty = prepare_features_for_regression(df, product, 'quantity')
            if X_qty is not None:
                trained_models_qty, predictions_qty, scores_qty, scaler_qty = train_ensemble_models(
                    X_qty, y_qty, product, 'quantity'
                )

                if trained_models_qty:
                    if product not in product_models:
                        product_models[product] = {}
                    product_models[product]['quantity'] = {
                        'models': trained_models_qty,
                        'scaler': scaler_qty,
                        'data': product_data_qty
                    }

                    # Predict future quantities
                    last_data_qty = X_qty.iloc[-1].values
                    future_predictions_qty = predict_future_regression(
                        trained_models_qty, scaler_qty, last_data_qty, steps=6, target_type='quantity'
                    )

                    # Ensure we have at least some predictions
                    if not future_predictions_qty:
                        st.warning(f"No predictions generated for {product} quantity. Skipping this product.")
                        continue

                    if future_predictions_qty:
                        ensemble_predictions_qty = []
                        for step in range(6):
                            step_predictions = [pred[step] for pred in future_predictions_qty.values() if
                                                len(pred) > step]
                            if step_predictions:
                                ensemble_predictions_qty.append(np.mean(step_predictions))

                        if product not in all_predictions:
                            all_predictions[product] = {}
                        all_predictions[product]['quantity'] = {
                            'individual': future_predictions_qty,
                            'ensemble': ensemble_predictions_qty
                        }

                        if product not in all_scores:
                            all_scores[product] = {}
                        all_scores[product]['quantity'] = scores_qty

                current_model += 1

        if target_variable in ["Both", "Total Sales Only"]:
            # Train total sales model
            X_sales, y_sales, product_data_sales = prepare_features_for_regression(df, product, 'total_sales')
            if X_sales is not None:
                trained_models_sales, predictions_sales, scores_sales, scaler_sales = train_ensemble_models(
                    X_sales, y_sales, product, 'total_sales'
                )

                if trained_models_sales:
                    if product not in product_models:
                        product_models[product] = {}
                    product_models[product]['total_sales'] = {
                        'models': trained_models_sales,
                        'scaler': scaler_sales,
                        'data': product_data_sales
                    }

                    # Predict future sales
                    last_data_sales = X_sales.iloc[-1].values
                    future_predictions_sales = predict_future_regression(
                        trained_models_sales, scaler_sales, last_data_sales, steps=6, target_type='total_sales'
                    )

                    # Ensure we have at least some predictions
                    if not future_predictions_sales:
                        st.warning(f"No predictions generated for {product} total sales. Skipping this product.")
                        continue

                    if future_predictions_sales:
                        ensemble_predictions_sales = []
                        for step in range(6):
                            step_predictions = [pred[step] for pred in future_predictions_sales.values() if
                                                len(pred) > step]
                            if step_predictions:
                                ensemble_predictions_sales.append(np.mean(step_predictions))

                        if product not in all_predictions:
                            all_predictions[product] = {}
                        all_predictions[product]['total_sales'] = {
                            'individual': future_predictions_sales,
                            'ensemble': ensemble_predictions_sales
                        }

                        if product not in all_scores:
                            all_scores[product] = {}
                        all_scores[product]['total_sales'] = scores_sales

                current_model += 1

    progress_bar.progress(1.0)
    status_text.text("✅ Advanced models trained successfully!")

    # Clear memory after training
    tf.keras.backend.clear_session()

    # Store results in session state
    st.session_state.product_models = product_models
    st.session_state.all_predictions = all_predictions
    st.session_state.all_scores = all_scores
    st.session_state.models_trained = True

    # Display results
    display_prediction_results(all_predictions, all_scores, show_ensemble, target_variable)


def train_hybrid_prediction_models(selected_products, target_variable, show_ensemble):
    """Train hybrid models combining XGBoost, LSTM, and SGD residual correction"""
    df = st.session_state.uploaded_data
    product_models = {}
    all_predictions = {}
    all_scores = {}

    total_models = len(selected_products)
    if target_variable == "Both":
        total_models *= 2  # Two models per product

    current_model = 0

    # Progress bar for hybrid training
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, product in enumerate(selected_products):
        status_text.text(f"🔮 Training hybrid models for {product}... (XGBoost + LSTM + SGD)")
        progress_bar.progress((i + 1) / len(selected_products))

        # Train models based on target variable selection
        if target_variable in ["Both", "Quantity Only"] and 'quantity' in df.columns:
            # Train quantity model
            X_qty, y_qty, product_data_qty = prepare_features_for_regression(df, product, 'quantity')
            if X_qty is not None:
                trained_models_qty, predictions_qty, scores_qty, scaler_qty = train_hybrid_models(
                    X_qty, y_qty, product, 'quantity'
                )

                if trained_models_qty:
                    if product not in product_models:
                        product_models[product] = {}
                    product_models[product]['quantity'] = {
                        'models': trained_models_qty,
                        'scaler': scaler_qty,
                        'data': product_data_qty
                    }

                    # Predict future quantity
                    last_data_qty = X_qty.iloc[-1].values
                    future_predictions_qty = predict_future_hybrid(
                        trained_models_qty, scaler_qty, last_data_qty, steps=6, target_type='quantity'
                    )

                    # Ensure we have at least some predictions
                    if not future_predictions_qty:
                        st.warning(f"No predictions generated for {product} quantity. Skipping this product.")
                        continue

                    if future_predictions_qty:
                        ensemble_predictions_qty = []
                        for step in range(6):
                            step_predictions = [pred[step] for pred in future_predictions_qty.values() if
                                                len(pred) > step]
                            if step_predictions:
                                ensemble_predictions_qty.append(np.mean(step_predictions))

                        if product not in all_predictions:
                            all_predictions[product] = {}
                        all_predictions[product]['quantity'] = {
                            'individual': future_predictions_qty,
                            'ensemble': ensemble_predictions_qty
                        }

                        if product not in all_scores:
                            all_scores[product] = {}
                        all_scores[product]['quantity'] = scores_qty

                current_model += 1

        if target_variable in ["Both", "Total Sales Only"]:
            # Train total sales model
            X_sales, y_sales, product_data_sales = prepare_features_for_regression(df, product, 'total_sales')
            if X_sales is not None:
                trained_models_sales, predictions_sales, scores_sales, scaler_sales = train_hybrid_models(
                    X_sales, y_sales, product, 'total_sales'
                )

                if trained_models_sales:
                    if product not in product_models:
                        product_models[product] = {}
                    product_models[product]['total_sales'] = {
                        'models': trained_models_sales,
                        'scaler': scaler_sales,
                        'data': product_data_sales
                    }

                    # Predict future sales
                    last_data_sales = X_sales.iloc[-1].values
                    future_predictions_sales = predict_future_hybrid(
                        trained_models_sales, scaler_sales, last_data_sales, steps=6, target_type='total_sales'
                    )

                    # Ensure we have at least some predictions
                    if not future_predictions_sales:
                        st.warning(f"No predictions generated for {product} total sales. Skipping this product.")
                        continue

                    if future_predictions_sales:
                        ensemble_predictions_sales = []
                        for step in range(6):
                            step_predictions = [pred[step] for pred in future_predictions_sales.values() if
                                                len(pred) > step]
                            if step_predictions:
                                ensemble_predictions_sales.append(np.mean(step_predictions))

                        if product not in all_predictions:
                            all_predictions[product] = {}
                        all_predictions[product]['total_sales'] = {
                            'individual': future_predictions_sales,
                            'ensemble': ensemble_predictions_sales
                        }

                        if product not in all_scores:
                            all_scores[product] = {}
                        all_scores[product]['total_sales'] = scores_sales

                current_model += 1

    progress_bar.progress(1.0)
    status_text.text("✅ Hybrid models trained successfully!")

    # Clear memory after training
    tf.keras.backend.clear_session()

    # Store results in session state
    st.session_state.product_models = product_models
    st.session_state.all_predictions = all_predictions
    st.session_state.all_scores = all_scores
    st.session_state.models_trained = True

    # Display results
    display_prediction_results(all_predictions, all_scores, show_ensemble, target_variable)


def train_hybrid_models(X, y, product, target_type):
    """Train hybrid models combining XGBoost, LSTM, and SGD residual correction"""
    trained_models = {}
    predictions = {}
    scores = {}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        # Train XGBoost model
        xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

        trained_models['XGBoost'] = xgb_model
        predictions['XGBoost'] = y_pred_xgb
        scores['XGBoost'] = {
            'MSE': mean_squared_error(y_test, y_pred_xgb),
            'MAE': mean_absolute_error(y_test, y_pred_xgb),
            'R2': r2_score(y_test, y_pred_xgb),
            'RMSE': xgb_rmse
        }

        # Train LSTM model
        # Limit data size for memory efficiency
        max_samples = min(1000, len(X_train_scaled))
        if len(X_train_scaled) > max_samples:
            indices = np.random.choice(len(X_train_scaled), max_samples, replace=False)
            X_train_scaled_lstm = X_train_scaled[indices]
            y_train_lstm = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        else:
            X_train_scaled_lstm = X_train_scaled
            y_train_lstm = y_train

        # Prepare LSTM data
        X_lstm = X_train_scaled_lstm.reshape((X_train_scaled_lstm.shape[0], X_train_scaled_lstm.shape[1], 1))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        # Create and train LSTM model
        lstm_model = Sequential([
            Input(shape=(X_lstm.shape[1], 1)),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])

        lstm_model.compile(optimizer=Adam(0.01), loss='mse', metrics=['mae'])

        # Train with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        lstm_model.fit(
            X_lstm, y_train_lstm,
            validation_data=(X_test_lstm, y_test),
            epochs=30,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0
        )

        y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lstm))

        trained_models['LSTM'] = lstm_model
        predictions['LSTM'] = y_pred_lstm
        scores['LSTM'] = {
            'MSE': mean_squared_error(y_test, y_pred_lstm),
            'MAE': mean_absolute_error(y_test, y_pred_lstm),
            'R2': r2_score(y_test, y_pred_lstm),
            'RMSE': lstm_rmse
        }

        # Build hybrid stacking with SGD residual correction
        # Use XGBoost as base predictor
        base_predictions = xgb_model.predict(X_train_scaled)
        residuals = y_train - base_predictions

        # Train SGD on residuals
        sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        sgd_model.fit(X_train_scaled, residuals)

        # Evaluate hybrid model
        base_pred_test = xgb_model.predict(X_test_scaled)
        residual_pred = sgd_model.predict(X_test_scaled)
        hybrid_predictions = base_pred_test + residual_pred

        hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_predictions))

        trained_models['Hybrid_XGB_SGD'] = {'xgb': xgb_model, 'sgd': sgd_model}
        predictions['Hybrid_XGB_SGD'] = hybrid_predictions
        scores['Hybrid_XGB_SGD'] = {
            'MSE': mean_squared_error(y_test, hybrid_predictions),
            'MAE': mean_absolute_error(y_test, hybrid_predictions),
            'R2': r2_score(y_test, hybrid_predictions),
            'RMSE': hybrid_rmse
        }

        # Models are kept in-memory (no saving to disk)

        st.success(f"✅ Hybrid models trained for {product} ({target_type})")
        st.info(f"XGBoost RMSE: {xgb_rmse:.2f}, LSTM RMSE: {lstm_rmse:.2f}, Hybrid RMSE: {hybrid_rmse:.2f}")

    except Exception as e:
        st.error(f"Error training hybrid models for {product} ({target_type}): {str(e)}")
        return {}, {}, {}, scaler

    return trained_models, predictions, scores, scaler


def predict_future_hybrid(trained_models, scaler, last_data, steps=6, target_type='total_sales'):
    """Predict future values using hybrid models"""
    future_predictions = {}

    try:
        # XGBoost predictions
        if 'XGBoost' in trained_models:
            xgb_model = trained_models['XGBoost']
            xgb_predictions = []
            current_data = last_data.copy()

            for step in range(steps):
                # Scale the current data
                current_scaled = scaler.transform(current_data.reshape(1, -1))
                pred = xgb_model.predict(current_scaled)[0]
                xgb_predictions.append(pred)

                # Update features for next prediction (simplified)
                if len(current_data) > 1:
                    current_data[0] += 1  # Increment year
                    current_data[-3] = pred  # Update lag feature

            future_predictions['XGBoost'] = xgb_predictions

        # LSTM predictions
        if 'LSTM' in trained_models:
            lstm_model = trained_models['LSTM']
            lstm_predictions = []
            current_data = last_data.copy()

            for step in range(steps):
                # Scale the current data
                current_scaled = scaler.transform(current_data.reshape(1, -1))
                current_lstm = current_scaled.reshape((1, current_scaled.shape[1], 1))
                pred = lstm_model.predict(current_lstm, verbose=0)[0][0]
                lstm_predictions.append(pred)

                # Update features for next prediction
                if len(current_data) > 1:
                    current_data[0] += 1  # Increment year
                    current_data[-3] = pred  # Update lag feature

            future_predictions['LSTM'] = lstm_predictions

        # Hybrid predictions
        if 'Hybrid_XGB_SGD' in trained_models:
            hybrid_models = trained_models['Hybrid_XGB_SGD']
            xgb_model = hybrid_models['xgb']
            sgd_model = hybrid_models['sgd']
            hybrid_predictions = []
            current_data = last_data.copy()

            for step in range(steps):
                # Scale the current data
                current_scaled = scaler.transform(current_data.reshape(1, -1))

                # Get base prediction from XGBoost
                base_pred = xgb_model.predict(current_scaled)[0]

                # Get residual correction from SGD
                residual_correction = sgd_model.predict(current_scaled)[0]

                # Combine predictions
                hybrid_pred = base_pred + residual_correction
                hybrid_predictions.append(hybrid_pred)

                # Update features for next prediction
                if len(current_data) > 1:
                    current_data[0] += 1  # Increment year
                    current_data[-3] = hybrid_pred  # Update lag feature

            future_predictions['Hybrid_XGB_SGD'] = hybrid_predictions

    except Exception as e:
        st.error(f"Error in hybrid prediction: {str(e)}")
        return {}

    return future_predictions


def reinforcement_learning_page():
    """Reinforcement learning for business optimization with feedback integration"""
    st.markdown('<h1 class="main-header">🤖 Reinforcement Learning Business Optimization</h1>', unsafe_allow_html=True)

    # Back button
    if st.button("← Back to Main Menu"):
        st.session_state.current_page = 'main'
        st.rerun()

    if st.session_state.uploaded_data is None:
        st.warning("Please upload sales data from the main menu first.")
        return

    df = st.session_state.uploaded_data

    st.markdown("## 🎯 Business Optimization with AI & Customer Feedback")
    st.markdown(
        "This page uses reinforcement learning and customer feedback to provide detailed business recommendations.")

    # Feedback data upload
    st.markdown("### 📊 Upload Customer Feedback Data")
    feedback_file = st.file_uploader(
        "Upload feedback.csv file",
        type=['csv'],
        help="Upload customer feedback data with columns: Product Category, Rating, Review"
    )

    feedback_data = None
    if feedback_file is not None:
        try:
            feedback_data = pd.read_csv(feedback_file)
            st.success(f"✅ Feedback data loaded successfully! Found {len(feedback_data)} reviews")

            # Display feedback summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reviews", len(feedback_data))
            with col2:
                st.metric("Average Rating", f"{feedback_data['Rating'].mean():.2f}")
            with col3:
                st.metric("Product Categories", feedback_data['Product Category'].nunique())

        except Exception as e:
            st.error(f"Error loading feedback file: {str(e)}")

    # Product selection for RL analysis
    products = df['product'].unique()
    selected_products = st.multiselect(
        "Choose products for business optimization",
        products,
        default=products[:min(3, len(products))],
        help="Select products to analyze with reinforcement learning and feedback"
    )

    if selected_products and st.button("🚀 Generate Advanced Business Recommendations", type="primary"):
        with st.spinner("🤖 Training reinforcement learning agent with feedback data..."):
            try:
                # Generate recommendations with feedback
                recommendations = generate_enhanced_business_recommendations(
                    df,
                    feedback_data,
                    st.session_state.get('product_models', {}),
                    st.session_state.get('all_predictions', {})
                )

                st.session_state.reinforcement_data = recommendations

                # Display enhanced recommendations
                display_enhanced_business_recommendations(recommendations, selected_products, feedback_data)

            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.info("Please ensure you have trained models in the Historical Sales Prediction page first.")


def display_enhanced_business_recommendations(recommendations, selected_products, feedback_data):
    """Display enhanced business recommendations with feedback analysis"""
    st.markdown("## 💡 Enhanced AI Business Recommendations")

    # Create tabs for different recommendation views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Recommendations", "📊 Performance Analysis", "📝 Feedback Analysis", "🚀 Action Plan"])

    with tab1:
        st.markdown("### Product-Specific Recommendations")

        for product in selected_products:
            if product in recommendations:
                rec = recommendations[product]

                with st.container():
                    st.markdown(f"### {product}")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Avg Sales", f"${rec['current_avg_sales']:.2f}")

                    with col2:
                        st.metric("Predicted Improvement", f"${rec['predicted_improvement']:.2f}")

                    with col3:
                        st.metric("Risk Level", rec['risk_level'])

                    # Feedback metrics if available
                    if rec.get('feedback_analysis'):
                        feedback = rec['feedback_analysis']
                        col4, col5, col6 = st.columns(3)

                        with col4:
                            st.metric("Avg Rating", f"{feedback.get('avg_rating', 0):.1f}/5")

                        with col5:
                            st.metric("Total Reviews", feedback.get('total_reviews', 0))

                        with col6:
                            positive_rate = (feedback.get('positive_reviews', 0) / feedback.get('total_reviews',
                                                                                                1)) * 100
                            st.metric("Positive Rate", f"{positive_rate:.1f}%")

                    # Recommendation card
                    st.markdown(f"""
                    <div class="reinforcement-card">
                        <h4>🎯 Recommended Action</h4>
                        <p><strong>{rec['recommended_action']}</strong></p>
                        <p>Confidence Score: {rec['confidence_score']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detailed suggestions
                    if rec.get('detailed_suggestions'):
                        st.markdown("#### 📋 Detailed Suggestions:")
                        for suggestion in rec['detailed_suggestions']:
                            st.markdown(f"- {suggestion}")

    with tab2:
        st.markdown("### Performance Analysis")

        if recommendations:
            # Create performance comparison chart
            products = list(recommendations.keys())
            current_sales = [recommendations[p]['current_avg_sales'] for p in products]
            predicted_improvements = [recommendations[p]['predicted_improvement'] for p in products]
            confidence_scores = [recommendations[p]['confidence_score'] for p in products]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=products,
                y=current_sales,
                name='Current Avg Sales',
                marker_color='lightblue'
            ))

            fig.add_trace(go.Bar(
                x=products,
                y=predicted_improvements,
                name='Predicted Improvement',
                marker_color='lightgreen'
            ))

            fig.update_layout(
                title="Current vs Predicted Performance",
                xaxis_title="Products",
                yaxis_title="Sales Amount ($)",
                barmode='group',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Confidence scores
            fig2 = px.bar(
                x=products,
                y=confidence_scores,
                title="AI Confidence Scores",
                labels={'x': 'Products', 'y': 'Confidence Score'}
            )

            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 📝 Feedback Analysis")

        if feedback_data is not None and recommendations:
            # Display feedback for all products
            products_with_feedback = [p for p in recommendations.keys()
                                      if p in feedback_data['Product Category'].unique()]

            if products_with_feedback:
                for selected_product_feedback in products_with_feedback:
                    st.markdown(f"#### 📊 Feedback Analysis for {selected_product_feedback}")

                    product_feedback = feedback_data[feedback_data['Product Category'] == selected_product_feedback]

                    if len(product_feedback) > 0:
                        # Rating distribution
                        fig = px.histogram(
                            product_feedback,
                            x='Rating',
                            title=f"Rating Distribution for {selected_product_feedback}",
                            nbins=5
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Sentiment analysis
                        sentiment = analyze_review_sentiment(product_feedback['Review'].tolist())

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Positive Reviews", sentiment.get('positive_count', 0))
                        with col2:
                            st.metric("Negative Reviews", sentiment.get('negative_count', 0))
                        with col3:
                            st.metric("Delivery Issues", sentiment.get('delivery_issues', 0))
                        with col4:
                            st.metric("Quality Issues", sentiment.get('quality_issues', 0))

                        # Sample reviews
                        st.markdown("#### 📝 Sample Reviews:")
                        sample_reviews = product_feedback.sample(min(5, len(product_feedback)))
                        for _, review in sample_reviews.iterrows():
                            rating_stars = "⭐" * int(review['Rating'])
                            st.markdown(f"**{rating_stars}** - {review['Review']}")

                        st.markdown("---")  # Add separator between products
            else:
                st.warning("No feedback data available for selected products.")
        else:
            st.info("Upload feedback data to see detailed feedback analysis.")

    with tab4:
        st.markdown("### 🚀 Action Plan")

        if recommendations:
            st.markdown("### Prioritized Action Items")

            # Sort recommendations by confidence score
            sorted_recs = sorted(recommendations.items(),
                                 key=lambda x: x[1]['confidence_score'],
                                 reverse=True)

            for i, (product, rec) in enumerate(sorted_recs, 1):
                st.markdown(f"""
                **{i}. {product}**
                - **Action**: {rec['recommended_action']}
                - **Expected Impact**: ${rec['predicted_improvement']:.2f} improvement
                - **Confidence**: {rec['confidence_score']:.2f}
                - **Risk Level**: {rec['risk_level']}
                """)

            # Summary metrics
            st.markdown("### 📊 Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                total_improvement = sum(rec['predicted_improvement'] for rec in recommendations.values())
                st.metric("Total Predicted Improvement", f"${total_improvement:.2f}")

            with col2:
                avg_confidence = np.mean([rec['confidence_score'] for rec in recommendations.values()])
                st.metric("Average Confidence", f"{avg_confidence:.2f}")

            with col3:
                high_risk_count = sum(1 for rec in recommendations.values() if rec['risk_level'] == 'High')
                st.metric("High Risk Products", high_risk_count)


def analytics_page():
    """Business analytics page"""
    st.markdown('<h1 class="main-header">📈 Business Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Back button
    if st.button("← Back to Main Menu"):
        st.session_state.current_page = 'main'
        st.rerun()

    if st.session_state.uploaded_data is None:
        st.warning("Please upload data from the main menu first.")
        return

    df = st.session_state.uploaded_data

    st.markdown("## 📊 Advanced Business Analytics")

    # Create tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "🎯 Product Analysis", "📊 Revenue Insights"])

    with tab1:
        st.markdown("### Sales Trends Analysis")

        # Monthly sales trend
        monthly_sales = df.groupby(df['date'].dt.to_period('M'))['total_sales'].sum()

        fig = px.line(
            x=monthly_sales.index.astype(str),
            y=monthly_sales.values,
            title="Monthly Sales Trend",
            labels={'x': 'Month', 'y': 'Total Sales'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Product performance over time
        product_monthly = df.groupby([df['date'].dt.to_period('M'), 'product'])['total_sales'].sum().reset_index()
        product_monthly['date'] = product_monthly['date'].astype(str)

        fig2 = px.line(
            product_monthly,
            x='date',
            y='total_sales',
            color='product',
            title="Product Sales Trends"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### Product Performance Analysis")

        # Top products by revenue
        top_products = df.groupby('product')['total_sales'].sum().sort_values(ascending=False)

        fig = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title="Top Products by Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Product sales distribution
        fig2 = px.box(
            df,
            x='product',
            y='total_sales',
            title="Sales Distribution by Product"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### Revenue Insights")

        # Revenue metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_revenue = df['total_sales'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.0f}")

        with col2:
            avg_order_value = df['total_sales'].mean()
            st.metric("Average Order Value", f"${avg_order_value:.2f}")

        with col3:
            total_orders = len(df)
            st.metric("Total Orders", total_orders)

        with col4:
            unique_customers = df['product'].nunique()  # Assuming product as proxy for customer segments
            st.metric("Product Categories", unique_customers)

        # Revenue by product category
        revenue_by_product = df.groupby('product')['total_sales'].agg(['sum', 'mean', 'count']).reset_index()
        revenue_by_product.columns = ['Product', 'Total Revenue', 'Average Order Value', 'Number of Orders']

        st.dataframe(revenue_by_product, use_container_width=True)


# Main application logic
def main():
    # Page routing
    if st.session_state.current_page == 'main':
        main_page()
    elif st.session_state.current_page == 'historical':
        historical_prediction_page()
    elif st.session_state.current_page == 'reinforcement':
        reinforcement_learning_page()
    elif st.session_state.current_page == 'analytics':
        analytics_page()


if __name__ == "__main__":
    main()

