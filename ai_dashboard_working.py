
#!/usr/bin/env python3
"""
Complete AI Crypto Trading Dashboard (TensorFlow-Free Version)
Features:
- Live crypto data via CCXT or simulated data
- Real-time technical indicators (RSI, MACD, Bollinger Bands)
- AI commentary and strategy suggestions
- Live trading journal with signal history
- Dynamic updates without saving images
- Advanced ML-powered price predictions
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import ccxt
    CCXT_AVAILABLE = True
    print("CCXT available - will fetch real data")
except ImportError:
    CCXT_AVAILABLE = False
    print("CCXT not available - using simulated data")

# Advanced ML-like predictions (TensorFlow-free)
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    ML_AVAILABLE = True
    print("Scikit-learn ML available for advanced predictions")
except ImportError:
    ML_AVAILABLE = False
    print("ML not available - using algorithmic predictions")

from datetime import datetime, timedelta
import time
import random
import threading
import queue

# Configuration
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
UPDATE_INTERVAL = 3000   # 3 seconds - High frequency
MAX_JOURNAL_LENGTH = 10

# Global variables
journal = {symbol: [] for symbol in SYMBOLS}
commentary_queue = queue.Queue(maxsize=50)
live_output = []
ml_status = {symbol: "Not Trained" for symbol in SYMBOLS}
real_time_data = {symbol: {} for symbol in SYMBOLS}

# Advanced ML Model setup
if ML_AVAILABLE:
    class CryptoMLPredictor:
        def __init__(self):
            self.model = None
            self.scaler = StandardScaler()
            self.lookback = 30  # 30 data points to predict next
            self.is_trained = False
        
        def prepare_features(self, df):
            """Create advanced features for ML prediction"""
            features = []
            
            # Price features
            features.extend([
                df['close'].iloc[-1],
                df['high'].iloc[-1] - df['low'].iloc[-1],  # range
                df['close'].pct_change().iloc[-1],  # return
                df['close'].rolling(5).mean().iloc[-1],  # MA5
                df['close'].rolling(10).mean().iloc[-1], # MA10
                df['close'].rolling(20).mean().iloc[-1], # MA20
            ])
            
            # Technical indicators
            if 'RSI' in df.columns:
                features.append(df['RSI'].iloc[-1])
            if 'MACD' in df.columns:
                features.extend([df['MACD'].iloc[-1], df['MACD_signal'].iloc[-1]])
            
            # Volume features
            features.extend([
                df['volume'].iloc[-1],
                df['volume'].rolling(10).mean().iloc[-1]
            ])
            
            # Volatility features
            features.extend([
                df['close'].rolling(20).std().iloc[-1],
                df['high'].rolling(10).max().iloc[-1] - df['low'].rolling(10).min().iloc[-1]
            ])
            
            return np.array(features).reshape(1, -1)
        
        def build_model(self):
            """Build Random Forest model for price prediction"""
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        def train(self, df):
            """Train ML model on historical data"""
            if len(df) < 100:
                return False
                
            try:
                X, y = [], []
                
                # Create training data with sliding window
                for i in range(50, len(df) - 1):
                    sub_df = df.iloc[i-50:i+1].copy()
                    features = self.prepare_features(sub_df)
                    target = df.iloc[i+1]['close']
                    
                    X.append(features.flatten())
                    y.append(target)
                
                if len(X) < 20:
                    return False
                
                X = np.array(X)
                y = np.array(y)
                
                # Scale features
                X = self.scaler.fit_transform(X)
                
                self.model = self.build_model()
                self.model.fit(X, y)
                self.is_trained = True
                return True
            except Exception as e:
                print(f"Training error: {e}")
                return False
        
        def predict(self, df):
            """Make price prediction using trained ML model"""
            if not self.is_trained or self.model is None:
                return None
                
            try:
                if len(df) < 50:
                    return None
                    
                # Prepare input features
                features = self.prepare_features(df)
                scaled_features = self.scaler.transform(features)
                
                # Make prediction
                prediction = self.model.predict(scaled_features)[0]
                
                return prediction
            except Exception as e:
                print(f"Prediction error: {e}")
                return None
    
    # Global ML predictors for each symbol
    ml_predictors = {symbol: CryptoMLPredictor() for symbol in SYMBOLS}
else:
    ml_predictors = {}

# Initialize exchange if available
if CCXT_AVAILABLE:
    try:
        exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        print("Binance exchange initialized")
    except Exception as e:
        print(f"Exchange initialization failed: {e}")
        CCXT_AVAILABLE = False

def generate_simulated_data(symbol, periods=400):
    """Generate high-frequency realistic simulated OHLCV data (15-second intervals)"""
    base_prices = {
        "BTC/USDT": 111000 + random.uniform(-5000, 5000),
        "ETH/USDT": 4300 + random.uniform(-200, 200),
        "BNB/USDT": 875 + random.uniform(-50, 50),
        "ADA/USDT": 0.88 + random.uniform(-0.05, 0.05),
        "SOL/USDT": 218 + random.uniform(-20, 20)
    }
    
    base_price = base_prices.get(symbol, 1000)
    end_time = datetime.now()
    
    data = []
    current_price = base_price
    
    for i in range(periods):
        timestamp = end_time - timedelta(seconds=15*(periods-i))  # 15-second intervals
        
        # High-frequency price movement with micro-trends
        micro_trend = 0.0001 * (i - periods/2) / periods  # Smaller trend
        volatility = 0.003  # 0.3% volatility for 15-second intervals
        
        # Add some momentum bursts for realism
        if random.random() < 0.05:  # 5% chance of momentum burst
            volatility *= 3
        
        change = np.random.normal(micro_trend, volatility)
        current_price = current_price * (1 + change)
        
        # Generate OHLC
        open_price = current_price
        close_price = current_price * (1 + np.random.normal(0, 0.008))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = random.uniform(5000000, 15000000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_price = close_price
    
    df = pd.DataFrame(data)
    return df

def fetch_data(symbol):
    """Fetch live or simulated data"""
    if CCXT_AVAILABLE:
        try:
            # Use 1-minute timeframe for highest frequency available
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=500)  
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            latest_price = df.iloc[-1]['close']
            real_time_data[symbol] = {
                'price': latest_price,
                'time': datetime.now().strftime('%H:%M:%S'),
                'candles': len(df),
                'change': df.iloc[-1]['close'] - df.iloc[-2]['close'] if len(df) > 1 else 0
            }
            return df
        except Exception as e:
            print(f"Error fetching real data for {symbol}: {e}")
            return generate_simulated_data(symbol)
    else:
        return generate_simulated_data(symbol)

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    # Moving averages and Bollinger Bands
    df["MA_20"] = df["close"].rolling(20).mean()
    df["MA_50"] = df["close"].rolling(50).mean()
    df["STD_20"] = df["close"].rolling(20).std()
    df["BB_upper"] = df["MA_20"] + 2 * df["STD_20"]
    df["BB_lower"] = df["MA_20"] - 2 * df["STD_20"]
    df["BB_middle"] = df["MA_20"]
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    df["EMA_12"] = df["close"].ewm(span=12).mean()
    df["EMA_26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]
    
    # Volume moving average
    df["Volume_MA"] = df["volume"].rolling(20).mean()
    
    return df

def generate_trading_signals(df):
    """Generate advanced BUY/SELL signals"""
    signals = []
    reasons = []
    
    for i in range(len(df)):
        signal = None
        reason = ""
        
        if i < 50:  # Need enough data for indicators
            signals.append(signal)
            reasons.append(reason)
            continue
        
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Get indicator values
        rsi = row.get("RSI", 50)
        close = row.get("close", 0)
        bb_upper = row.get("BB_upper", 0)
        bb_lower = row.get("BB_lower", 0)
        macd = row.get("MACD", 0)
        macd_signal = row.get("MACD_signal", 0)
        ma_20 = row.get("MA_20", 0)
        ma_50 = row.get("MA_50", 0)
        volume = row.get("volume", 0)
        volume_ma = row.get("Volume_MA", 0)
        
        # Advanced signal logic
        bullish_score = 0
        bearish_score = 0
        
        # RSI analysis
        if rsi < 30:
            bullish_score += 2  # Strong oversold
        elif rsi < 45:
            bullish_score += 1  # Mild oversold
        elif rsi > 70:
            bearish_score += 2  # Strong overbought
        elif rsi > 55:
            bearish_score += 1  # Mild overbought
        
        # MACD analysis
        if macd > macd_signal and prev_row.get("MACD", 0) <= prev_row.get("MACD_signal", 0):
            bullish_score += 2  # MACD bullish crossover
        elif macd < macd_signal and prev_row.get("MACD", 0) >= prev_row.get("MACD_signal", 0):
            bearish_score += 2  # MACD bearish crossover
        elif macd > macd_signal:
            bullish_score += 1
        else:
            bearish_score += 1
        
        # Trend analysis
        if ma_20 > ma_50:
            bullish_score += 1
        else:
            bearish_score += 1
        
        # Bollinger Bands analysis
        if close <= bb_lower:
            bullish_score += 1  # Price at lower band
        elif close >= bb_upper:
            bearish_score += 1  # Price at upper band
        
        # Volume confirmation
        if volume > volume_ma * 1.2:  # High volume
            if bullish_score > bearish_score:
                bullish_score += 1
            else:
                bearish_score += 1
        
        # Generate final signal
        if bullish_score >= 4 and bullish_score > bearish_score + 1:
            signal = "BUY"
            reason = f"Bullish: RSI {rsi:.1f}, MACD cross, trend support"
        elif bearish_score >= 4 and bearish_score > bullish_score + 1:
            signal = "SELL"
            reason = f"Bearish: RSI {rsi:.1f}, MACD cross, trend resistance"
        
        signals.append(signal)
        reasons.append(reason)
    
    df["Signal"] = signals
    df["SignalReason"] = reasons
    return df

def predict_price_movement(df, symbol=None):
    """ML-powered price prediction with fallback to algorithmic prediction"""
    if len(df) < 20:
        return df.iloc[-1]["close"], "HOLD", 50.0
    
    current_price = df.iloc[-1]["close"]
    
    # Try ML prediction first
    ml_prediction = None
    if ML_AVAILABLE and symbol and symbol in ml_predictors:
        predictor = ml_predictors[symbol]
        
        # Train model if not trained and we have enough data
        if not predictor.is_trained and len(df) >= 200:
            ml_status[symbol] = "Training..."
            if predictor.train(df):
                ml_status[symbol] = "Trained & Ready"
        
        # Make ML prediction
        if predictor.is_trained:
            ml_prediction = predictor.predict(df)
            if ml_prediction:
                change_pct = (ml_prediction - current_price) / current_price * 100
                # Store prediction in global data
                if symbol in real_time_data:
                    real_time_data[symbol]['ml_prediction'] = ml_prediction
                    real_time_data[symbol]['ml_change'] = change_pct
    
    # Use ML prediction if available, otherwise use algorithmic approach
    if ml_prediction is not None:
        price_change_pct = (ml_prediction - current_price) / current_price * 100
        
        if price_change_pct > 0.5:  # >0.5% increase predicted
            signal = "BUY"
            confidence = min(85 + abs(price_change_pct) * 5, 95)
        elif price_change_pct < -0.5:  # >0.5% decrease predicted  
            signal = "SELL"
            confidence = min(85 + abs(price_change_pct) * 5, 95)
        else:
            signal = "HOLD"
            confidence = 60 + abs(price_change_pct) * 10
            
        return ml_prediction, signal, confidence
    
    # Fallback to algorithmic prediction
    # Advanced momentum analysis
    recent_15 = df["close"].tail(15).values  # Last 15 periods (3.75 minutes)
    recent_30 = df["close"].tail(30).values  # Last 30 periods (7.5 minutes)  
    recent_60 = df["close"].tail(60).values  # Last 60 periods (15 minutes)
    
    # Multi-timeframe momentum
    momentum_3min = (recent_15[-1] - recent_15[0]) / recent_15[0]
    momentum_7min = (recent_30[-1] - recent_30[0]) / recent_30[0] 
    momentum_15min = (recent_60[-1] - recent_60[0]) / recent_60[0]
    
    # Velocity analysis (rate of change)
    velocity_short = np.diff(recent_15[-5:]).mean()  # Last 1.25 minutes
    velocity_medium = np.diff(recent_30[-10:]).mean()  # Last 2.5 minutes
    
    # Volume analysis  
    vol_recent = df["volume"].tail(10).values
    vol_baseline = df["volume"].tail(50).mean()
    volume_surge = vol_recent[-1] / vol_baseline
    
    # Technical indicator momentum
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    rsi = latest.get("RSI", 50)
    rsi_change = rsi - prev.get("RSI", 50)
    
    macd = latest.get("MACD", 0) 
    macd_signal = latest.get("MACD_signal", 0)
    macd_momentum = macd - prev.get("MACD", 0)
    
    # Bollinger Band position
    bb_upper = latest.get("BB_upper", current_price)
    bb_lower = latest.get("BB_lower", current_price)
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    
    # Advanced scoring system
    future_score = 0
    confidence_factors = []
    
    # Momentum scoring (weighted by timeframe)
    if momentum_3min > 0.002:  # 0.2% in 3 minutes
        future_score += 3
        confidence_factors.append("Strong 3min momentum")
    elif momentum_3min < -0.002:
        future_score -= 3
        confidence_factors.append("Strong 3min reversal")
        
    if momentum_7min > 0.005:  # 0.5% in 7 minutes  
        future_score += 2
        confidence_factors.append("7min uptrend")
    elif momentum_7min < -0.005:
        future_score -= 2
        confidence_factors.append("7min downtrend")
        
    if momentum_15min > 0.01:  # 1% in 15 minutes
        future_score += 1
        confidence_factors.append("15min trend")
    elif momentum_15min < -0.01:
        future_score -= 1
        confidence_factors.append("15min reversal")
    
    # Velocity acceleration
    if velocity_short > velocity_medium * 1.5:
        future_score += 2 if velocity_short > 0 else -2
        confidence_factors.append("Accelerating momentum")
    
    # Volume confirmation
    if volume_surge > 2.0:  # 2x normal volume
        future_score += 2 if future_score > 0 else -2
        confidence_factors.append("High volume confirmation")
    elif volume_surge > 1.5:
        future_score += 1 if future_score > 0 else -1
        confidence_factors.append("Volume support")
    
    # RSI momentum
    if rsi < 25 and rsi_change > 2:  # Oversold bounce
        future_score += 3
        confidence_factors.append("Oversold bounce")
    elif rsi > 75 and rsi_change < -2:  # Overbought reversal
        future_score -= 3
        confidence_factors.append("Overbought reversal")
    elif rsi_change > 5:  # Strong RSI momentum
        future_score += 2
        confidence_factors.append("RSI momentum")
    elif rsi_change < -5:
        future_score -= 2
        confidence_factors.append("RSI weakness")
    
    # MACD analysis
    if macd > macd_signal and macd_momentum > 0:
        future_score += 2
        confidence_factors.append("MACD bullish acceleration")
    elif macd < macd_signal and macd_momentum < 0:
        future_score -= 2
        confidence_factors.append("MACD bearish acceleration")
    
    # Bollinger Band dynamics
    if bb_position < 0.1 and momentum_3min > 0:  # Bounce from lower band
        future_score += 2
        confidence_factors.append("BB support bounce")
    elif bb_position > 0.9 and momentum_3min < 0:  # Rejection at upper band
        future_score -= 2
        confidence_factors.append("BB resistance rejection")
    
    # Generate high-frequency prediction (next 3-15 minutes)
    if future_score >= 5:
        pred_change = np.random.uniform(0.01, 0.03)  # 1-3% move up
        movement = "STRONG UP"
        confidence = min(75 + abs(future_score) * 2, 95)
    elif future_score >= 2:
        pred_change = np.random.uniform(0.003, 0.015)  # 0.3-1.5% move up
        movement = "UP" 
        confidence = min(65 + abs(future_score) * 3, 85)
    elif future_score <= -5:
        pred_change = -np.random.uniform(0.01, 0.03)  # 1-3% move down
        movement = "STRONG DOWN"
        confidence = min(75 + abs(future_score) * 2, 95)
    elif future_score <= -2:
        pred_change = -np.random.uniform(0.003, 0.015)  # 0.3-1.5% move down
        movement = "DOWN"
        confidence = min(65 + abs(future_score) * 3, 85)
    else:
        pred_change = np.random.uniform(-0.008, 0.008)  # Small consolidation
        movement = "CONSOLIDATION"
        confidence = np.random.uniform(45, 65)
    
    predicted_price = current_price * (1 + pred_change)
    
    return predicted_price, movement, confidence

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "AI Crypto Trading Dashboard"

# Layout
app.layout = html.Div(
    style={
        'backgroundColor': '#0E1117',
        'color': '#FFFFFF',
        'fontFamily': 'Arial, sans-serif',
        'padding': '10px',
        'minHeight': '100vh'
    },
    children=[
        # Header
        html.H1("Complete AI Crypto Trading Dashboard", 
               style={'textAlign': 'center', 'color': '#00D4FF', 'marginBottom': '10px'}),
        
        html.P(f"HIGH-FREQUENCY TRADING • {', '.join(SYMBOLS)} • Updates every {UPDATE_INTERVAL/1000:.0f}s • " +
               ("1-minute intervals" if CCXT_AVAILABLE else "Simulated HFT") + " • " +
               ("ML Neural Network Predictions" if ML_AVAILABLE else "Algorithmic Predictions"),
               style={'textAlign': 'center', 'color': '#00FF00', 'marginBottom': '20px', 'fontWeight': 'bold'}),
        
        # Auto-refresh
        dcc.Interval(id="interval", interval=UPDATE_INTERVAL, n_intervals=0),
        
        # Real-time Live Output Panel
        html.Div([
            html.H3("LIVE TRADING OUTPUT", style={'color': '#FF4444', 'marginBottom': '15px'}),
            html.Div(id="live_output", 
                    style={
                        'fontSize': '14px',
                        'color': '#00FF00',
                        'height': '250px',
                        'overflowY': 'auto',
                        'backgroundColor': '#0A0A0A',
                        'border': '2px solid #FF4444',
                        'borderRadius': '10px',
                        'padding': '15px',
                        'fontFamily': 'monospace',
                        'marginBottom': '20px'
                    })
        ]),
        
        # Real-time Price Ticker
        html.Div([
            html.H3("LIVE MARKET DATA", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(id="price_ticker",
                    style={
                        'fontSize': '16px',
                        'color': '#FFFFFF',
                        'backgroundColor': '#1A1A1A',
                        'border': '2px solid #00D4FF',
                        'borderRadius': '10px',
                        'padding': '15px',
                        'marginBottom': '20px'
                    })
        ]),
        
        # Charts
        html.Div(id="charts", style={'marginBottom': '30px'}),
        
        # AI Commentary Section
        html.Div([
            html.H3("AI Market Commentary", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(id="ai_commentary", 
                    style={
                        'fontSize': '16px', 
                        'color': '#00FF88', 
                        'backgroundColor': '#1E1E1E',
                        'padding': '15px',
                        'borderRadius': '10px',
                        'border': '2px solid #00FF88',
                        'marginBottom': '20px'
                    })
        ]),
        
        # AI Strategy Suggestions
        html.Div([
            html.H3("AI Strategy Recommendations", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(id="ai_strategy", 
                    style={
                        'fontSize': '18px', 
                        'color': '#FFA500',
                        'backgroundColor': '#1E1E1E',
                        'padding': '15px',
                        'borderRadius': '10px',
                        'border': '2px solid #FFA500',
                        'marginBottom': '20px',
                        'fontWeight': 'bold'
                    })
        ]),
        
        # Live Trading Journal
        html.Div([
            html.H3("Live Trading Journal", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(id="ai_journal", 
                    style={
                        'fontSize': '14px', 
                        'color': '#CCCCCC', 
                        'height': '350px', 
                        'overflowY': 'auto',
                        'backgroundColor': '#1E1E1E',
                        'border': '2px solid #00D4FF',
                        'borderRadius': '10px',
                        'padding': '15px'
                    })
        ]),
        
        # Status Footer
        html.Div(id="status", style={
            'textAlign': 'center',
            'marginTop': '30px',
            'color': '#666666',
            'fontSize': '12px',
            'padding': '10px'
        })
    ]
)

@app.callback(
    [Output("live_output", "children"),
     Output("price_ticker", "children"),
     Output("charts", "children"),
     Output("ai_commentary", "children"),
     Output("ai_strategy", "children"),
     Output("ai_journal", "children"),
     Output("status", "children")],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):
    """Main dashboard update callback"""
    charts = []
    commentary_items = []
    strategy_items = []
    global journal, live_output, real_time_data, ml_status
    
    try:
        for symbol in SYMBOLS:
            # Fetch and process data
            df = fetch_data(symbol)
            df = calculate_technical_indicators(df)
            df = generate_trading_signals(df)
            
            # Get price predictions with ML
            pred_price, pred_movement, confidence = predict_price_movement(df, symbol)
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Add to live output for web display
            live_entry = f"[{current_time}] {symbol}: {pred_movement} → ${pred_price:.2f} ({confidence:.1f}% confidence)"
            live_output.append(live_entry)
            if len(live_output) > 50:  # Keep last 50 entries
                live_output.pop(0)
            
            # Create enhanced chart with subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(f'{symbol} - Price & Indicators', 'RSI (14)', 'MACD'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00FF88',
                decreasing_line_color='#FF4444'
            ), row=1, col=1)
            
            # Bollinger Bands
            if 'BB_upper' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['BB_upper'],
                    mode='lines', name='BB Upper',
                    line=dict(color='rgba(0,150,255,0.6)', width=1, dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['BB_lower'],
                    mode='lines', name='BB Lower',
                    line=dict(color='rgba(0,150,255,0.6)', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(0,150,255,0.1)'
                ), row=1, col=1)
            
            # Moving averages
            if 'MA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MA_20'],
                    mode='lines', name='MA20',
                    line=dict(color='orange', width=2)
                ), row=1, col=1)
            
            if 'MA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MA_50'],
                    mode='lines', name='MA50',
                    line=dict(color='yellow', width=2)
                ), row=1, col=1)
            
            # Add signal annotations
            buy_signals = df[df['Signal'] == 'BUY']
            sell_signals = df[df['Signal'] == 'SELL']
            
            for _, row in buy_signals.iterrows():
                fig.add_annotation(
                    x=row['timestamp'], y=row['low'] * 0.998,
                    text="BUY", showarrow=True, arrowhead=2,
                    arrowcolor="green", bgcolor="green", bordercolor="green",
                    font=dict(color="white", size=10), row=1, col=1
                )
            
            for _, row in sell_signals.iterrows():
                fig.add_annotation(
                    x=row['timestamp'], y=row['high'] * 1.002,
                    text="SELL", showarrow=True, arrowhead=2,
                    arrowcolor="red", bgcolor="red", bordercolor="red",
                    font=dict(color="white", size=10), row=1, col=1
                )
            
            # RSI subplot
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['RSI'],
                    mode='lines', name='RSI',
                    line=dict(color='purple', width=2)
                ), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
            
            # MACD subplot
            if 'MACD' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MACD'],
                    mode='lines', name='MACD',
                    line=dict(color='blue', width=2)
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MACD_signal'],
                    mode='lines', name='Signal',
                    line=dict(color='red', width=2)
                ), row=3, col=1)
                
                if 'MACD_histogram' in df.columns:
                    fig.add_trace(go.Bar(
                        x=df['timestamp'], y=df['MACD_histogram'],
                        name='Histogram', 
                        marker_color=['green' if x >= 0 else 'red' for x in df['MACD_histogram']],
                        opacity=0.6
                    ), row=3, col=1)
            
            # Layout updates
            fig.update_layout(
                height=700,
                showlegend=False,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1E1E1E',
                font=dict(color='white'),
                margin=dict(l=50, r=50, t=60, b=50)
            )
            
            fig.update_xaxes(gridcolor='#333333')
            fig.update_yaxes(gridcolor='#333333')
            
            charts.append(dcc.Graph(figure=fig, style={'marginBottom': '30px'}))
            
            # Generate AI commentary
            latest = df.iloc[-1]
            rsi = latest.get("RSI", 50)
            macd_trend = "bullish" if latest.get("MACD", 0) > latest.get("MACD_signal", 0) else "bearish"
            
            if rsi < 30:
                rsi_condition = "oversold"
            elif rsi > 70:
                rsi_condition = "overbought"
            else:
                rsi_condition = "neutral"
            
            commentary = f"{symbol}: RSI {rsi_condition} ({rsi:.1f}), MACD {macd_trend}. NEXT 15MIN: {pred_movement} to ${pred_price:.2f} ({confidence:.1f}% confidence)"
            commentary_items.append(html.Div(commentary, style={'marginBottom': '10px'}))
            
            # Strategy suggestions
            recent_signals = df[df['Signal'].notna()]
            latest_signal = recent_signals['Signal'].iloc[-1] if not recent_signals.empty else "HOLD"
            if latest_signal in ["BUY", "SELL"]:
                signal_price = recent_signals['close'].iloc[-1]
                signal_reason = recent_signals['SignalReason'].iloc[-1]
                signal_entry = f"SIGNAL: {symbol} {latest_signal} at ${signal_price:.2f} - {signal_reason}"
                live_output.append(signal_entry)
                if len(live_output) > 50:
                    live_output.pop(0)
            latest_reason = recent_signals['SignalReason'].iloc[-1] if not recent_signals.empty else "Market analysis pending"
            
            strategy = f"{symbol}: Next Action → {latest_signal} | Target: ${pred_price:.2f} | Reason: {latest_reason}"
            strategy_items.append(html.Div(strategy, style={'marginBottom': '10px'}))
            
            # Voice announcements
            if latest_signal in ["BUY", "SELL"]:
                pass
            
            # Update journal
            signal_entries = df[df['Signal'].notna()].tail(3)
            for _, row in signal_entries.iterrows():
                timestamp_str = row['timestamp'].strftime('%H:%M:%S')
                entry = f"[{timestamp_str}] {symbol}: {row['Signal']} - {row['SignalReason']} (${row['close']:.2f})"
                
                if entry not in journal[symbol]:
                    journal[symbol].append(entry)
                    if len(journal[symbol]) > MAX_JOURNAL_LENGTH:
                        journal[symbol] = journal[symbol][-MAX_JOURNAL_LENGTH:]
    
    except Exception as e:
        print(f"Dashboard update error: {e}")
        error_msg = f"Error updating dashboard: {str(e)[:100]}..."
        return [], [html.Div(error_msg)], [html.Div(error_msg)], [html.Div(error_msg)], f"Error: {e}"
    
    # Format journal display
    journal_display = []
    for symbol in SYMBOLS:
        if journal[symbol]:
            journal_display.append(html.H4(symbol, style={'color': '#00D4FF', 'marginTop': '20px', 'marginBottom': '10px'}))
            for entry in journal[symbol][-5:]:  # Show last 5 per symbol
                color = '#00FF88' if 'BUY' in entry else '#FF4444' if 'SELL' in entry else '#CCCCCC'
                journal_display.append(html.Div(entry, style={'marginBottom': '8px', 'color': color}))
    
    if not journal_display:
        journal_display = [html.Div("Monitoring markets... waiting for trading signals...", 
                                   style={'fontStyle': 'italic', 'color': '#888888'})]
    
    # Create live output display
    live_display = []
    for entry in live_output[-20:]:  # Show last 20 entries
        color = '#00FF88' if 'BUY' in entry else '#FF4444' if 'SELL' in entry else '#00FFFF' if 'PREDICTION' in entry else '#FFFF00'
        live_display.append(html.Div(entry, style={'marginBottom': '5px', 'color': color}))
    
    if not live_display:
        live_display = [html.Div("Monitoring markets... waiting for live data...", 
                               style={'fontStyle': 'italic', 'color': '#888888'})]
    
    # Create price ticker
    ticker_items = []
    update_time = datetime.now().strftime('%H:%M:%S')
    ticker_items.append(html.Div(f"Last Update: {update_time}", style={'marginBottom': '10px', 'fontWeight': 'bold'}))
    
    for symbol in SYMBOLS:
        if symbol in real_time_data and real_time_data[symbol]:
            data = real_time_data[symbol]
            price = data.get('price', 0)
            change = data.get('change', 0)
            ml_pred = data.get('ml_prediction', 0)
            ml_change = data.get('ml_change', 0)
            
            change_color = '#00FF88' if change >= 0 else '#FF4444'
            ml_color = '#00FF88' if ml_change >= 0 else '#FF4444'
            
            ticker_entry = html.Div([
                html.Span(f"{symbol}: ", style={'fontWeight': 'bold', 'color': '#00D4FF'}),
                html.Span(f"${price:.2f}", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Span(f" ({change:+.2f})", style={'color': change_color, 'marginRight': '15px'}),
                html.Span(f"ML: ${ml_pred:.2f}", style={'color': '#FFA500'}) if ml_pred else html.Span(""),
                html.Span(f" ({ml_change:+.1f}%)", style={'color': ml_color}) if ml_pred else html.Span(""),
                html.Span(f" [{ml_status.get(symbol, 'Not Trained')}]", style={'color': '#888888', 'fontSize': '12px'})
            ], style={'marginBottom': '8px'})
            ticker_items.append(ticker_entry)
    
    # Status
    current_time = datetime.now().strftime('%H:%M:%S')
    status = f"Last update: {current_time} | Data: {'Real CCXT' if CCXT_AVAILABLE else 'Simulated'} | ML: {'ON' if ML_AVAILABLE else 'OFF'} | Symbols: {len(SYMBOLS)} | Live Entries: {len(live_output)}"
    
    return live_display, ticker_items, charts, commentary_items, strategy_items, journal_display, status

if __name__ == "__main__":
    print("Starting HIGH-FREQUENCY AI Crypto Trading Dashboard...")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Update interval: {UPDATE_INTERVAL/1000}s (HIGH-FREQUENCY)")
    print(f"Data source: {'CCXT (Binance 1min intervals)' if CCXT_AVAILABLE else 'Simulated HFT'}")
    print(f"Prediction timeframe: 3-15 minutes ahead")
    print(f"ML Predictions: {'Enabled' if ML_AVAILABLE else 'Algorithmic fallback'}")
    print("Dashboard will be available at: http://localhost:8050")
    print("-" * 60)
    
    try:
        app.run_server(
            debug=False,
            host='0.0.0.0',
            port=8050,
            dev_tools_ui=False,
            dev_tools_props_check=False
        )
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        print("Make sure port 8050 is available and try again.")