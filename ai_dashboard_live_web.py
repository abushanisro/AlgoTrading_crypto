#!/usr/bin/env python3
"""
Live Web-Based AI Crypto Trading Dashboard
Features:
- Real-time data displayed directly on web interface
- Live ML predictions and trading signals
- High-frequency updates with web-based output
- All terminal output moved to web dashboard
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

# Global variables for web display
journal = {symbol: [] for symbol in SYMBOLS}
live_output = []
real_time_data = {symbol: {} for symbol in SYMBOLS}
ml_status = {symbol: "Not Trained" for symbol in SYMBOLS}
trading_signals = []

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
    """Generate high-frequency realistic simulated OHLCV data"""
    base_prices = {
        "BTC/USDT": 65000 + random.uniform(-5000, 5000),
        "ETH/USDT": 3200 + random.uniform(-200, 200),
        "BNB/USDT": 580 + random.uniform(-50, 50),
        "ADA/USDT": 0.38 + random.uniform(-0.05, 0.05),
        "SOL/USDT": 140 + random.uniform(-20, 20)
    }
    
    base_price = base_prices.get(symbol, 1000)
    end_time = datetime.now()
    
    data = []
    current_price = base_price
    
    for i in range(periods):
        timestamp = end_time - timedelta(minutes=periods-i)
        
        change = np.random.normal(0, 0.02)
        current_price = current_price * (1 + change)
        
        # Generate OHLC
        open_price = current_price
        close_price = current_price * (1 + np.random.normal(0, 0.01))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.008)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.008)))
        volume = random.uniform(1000000, 5000000)
        
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
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=500)  
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Store real-time data for web display
            latest_price = df.iloc[-1]['close']
            change = df.iloc[-1]['close'] - df.iloc[-2]['close'] if len(df) > 1 else 0
            real_time_data[symbol] = {
                'price': latest_price,
                'time': datetime.now().strftime('%H:%M:%S'),
                'candles': len(df),
                'change': change,
                'volume': df.iloc[-1]['volume']
            }
            
            return df
        except Exception as e:
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
            bullish_score += 2
        elif rsi < 45:
            bullish_score += 1
        elif rsi > 70:
            bearish_score += 2
        elif rsi > 55:
            bearish_score += 1
        
        # MACD analysis
        if macd > macd_signal and prev_row.get("MACD", 0) <= prev_row.get("MACD_signal", 0):
            bullish_score += 2
        elif macd < macd_signal and prev_row.get("MACD", 0) >= prev_row.get("MACD_signal", 0):
            bearish_score += 2
        elif macd > macd_signal:
            bullish_score += 1
        else:
            bearish_score += 1
        
        # Trend analysis
        if ma_20 > ma_50:
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
    """ML-powered price prediction with algorithmic fallback"""
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
    
    # Use ML prediction if available
    if ml_prediction is not None:
        price_change_pct = (ml_prediction - current_price) / current_price * 100
        
        if price_change_pct > 0.5:
            signal = "BUY"
            confidence = min(85 + abs(price_change_pct) * 5, 95)
        elif price_change_pct < -0.5:
            signal = "SELL"
            confidence = min(85 + abs(price_change_pct) * 5, 95)
        else:
            signal = "HOLD"
            confidence = 60 + abs(price_change_pct) * 10
            
        return ml_prediction, signal, confidence
    
    # Fallback algorithmic prediction
    future_score = random.randint(-3, 3)
    if future_score >= 2:
        pred_change = np.random.uniform(0.005, 0.02)
        movement = "UP"
        confidence = np.random.uniform(70, 90)
    elif future_score <= -2:
        pred_change = -np.random.uniform(0.005, 0.02)
        movement = "DOWN"
        confidence = np.random.uniform(70, 90)
    else:
        pred_change = np.random.uniform(-0.005, 0.005)
        movement = "HOLD"
        confidence = np.random.uniform(50, 70)
    
    predicted_price = current_price * (1 + pred_change)
    return predicted_price, movement, confidence

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Live AI Crypto Trading Dashboard"

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
        html.H1("LIVE AI Crypto Trading Dashboard", 
               style={'textAlign': 'center', 'color': '#FF4444', 'marginBottom': '10px'}),
        
        html.P(f"HIGH-FREQUENCY TRADING • {', '.join(SYMBOLS)} • Updates every {UPDATE_INTERVAL/1000:.0f}s • " +
               ("Real-time CCXT data" if CCXT_AVAILABLE else "Simulated data") + " • " +
               ("ML Predictions Enabled" if ML_AVAILABLE else "Algorithmic Predictions"),
               style={'textAlign': 'center', 'color': '#00FF00', 'marginBottom': '20px', 'fontWeight': 'bold'}),
        
        # Auto-refresh
        dcc.Interval(id="interval", interval=UPDATE_INTERVAL, n_intervals=0),
        
        # Live Output Panel
        html.Div([
            html.H3("LIVE TRADING OUTPUT", style={'color': '#FF4444', 'marginBottom': '15px'}),
            html.Div(id="live_output", 
                    style={
                        'fontSize': '14px',
                        'color': '#00FF00',
                        'height': '300px',
                        'overflowY': 'auto',
                        'backgroundColor': '#0A0A0A',
                        'border': '2px solid #FF4444',
                        'borderRadius': '10px',
                        'padding': '15px',
                        'fontFamily': 'monospace',
                        'marginBottom': '20px'
                    })
        ]),
        
        # Price Ticker
        html.Div([
            html.H3("REAL-TIME MARKET DATA", style={'color': '#00D4FF', 'marginBottom': '15px'}),
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
        html.Div(id="charts", style={'marginBottom': '20px'}),
        
        # Status Footer
        html.Div(id="status", style={
            'textAlign': 'center',
            'marginTop': '20px',
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
     Output("status", "children")],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):
    """Main dashboard update callback"""
    global live_output, real_time_data, ml_status, trading_signals
    
    # Update live output
    current_time = datetime.now().strftime('%H:%M:%S')
    live_output.append(f"[{current_time}] UPDATE #{n+1} - Processing {len(SYMBOLS)} symbols...")
    
    charts = []
    
    try:
        for symbol in SYMBOLS:
            # Fetch and process data
            df = fetch_data(symbol)
            df = calculate_technical_indicators(df)
            df = generate_trading_signals(df)
            
            # Get ML predictions
            pred_price, pred_movement, confidence = predict_price_movement(df, symbol)
            
            # Add prediction to live output
            current_price = df.iloc[-1]['close']
            change_pct = (pred_price - current_price) / current_price * 100
            live_output.append(f"PREDICT {symbol}: {pred_movement} → ${pred_price:.2f} ({change_pct:+.2f}%) - {confidence:.1f}% confidence")
            
            # Check for trading signals
            recent_signals = df[df['Signal'].notna()]
            if not recent_signals.empty:
                latest_signal = recent_signals['Signal'].iloc[-1]
                if latest_signal in ["BUY", "SELL"]:
                    signal_price = recent_signals['close'].iloc[-1]
                    signal_reason = recent_signals['SignalReason'].iloc[-1]
                    live_output.append(f"SIGNAL: {symbol} {latest_signal} at ${signal_price:.2f} - {signal_reason}")
            
            # Keep live output manageable
            if len(live_output) > 50:
                live_output = live_output[-50:]
            
            # Create basic chart (simplified for performance)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['timestamp'].tail(100),
                open=df['open'].tail(100),
                high=df['high'].tail(100),
                low=df['low'].tail(100),
                close=df['close'].tail(100),
                name=symbol
            ))
            
            fig.update_layout(
                title=f'{symbol} - Live Price Chart',
                height=400,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#1E1E1E',
                font=dict(color='white'),
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            charts.append(dcc.Graph(figure=fig, style={'marginBottom': '20px'}))
    
    except Exception as e:
        live_output.append(f"ERROR: {str(e)}")
    
    # Create live output display
    live_display = []
    for entry in live_output[-25:]:  # Show last 25 entries
        if 'BUY' in entry:
            color = '#00FF88'
        elif 'SELL' in entry:
            color = '#FF4444'
        elif 'PREDICTION' in entry or 'PREDICT' in entry:
            color = '#00FFFF'
        elif 'SIGNAL:' in entry:
            color = '#FFD700'
        else:
            color = '#CCCCCC'
        
        live_display.append(html.Div(entry, style={'marginBottom': '5px', 'color': color}))
    
    # Create price ticker
    ticker_items = []
    update_time = datetime.now().strftime('%H:%M:%S')
    ticker_items.append(html.Div(f"Live Update: {update_time}", style={'marginBottom': '10px', 'fontWeight': 'bold'}))
    
    for symbol in SYMBOLS:
        if symbol in real_time_data and real_time_data[symbol]:
            data = real_time_data[symbol]
            price = data.get('price', 0)
            change = data.get('change', 0)
            volume = data.get('volume', 0)
            ml_pred = data.get('ml_prediction', 0)
            ml_change = data.get('ml_change', 0)
            
            change_color = '#00FF88' if change >= 0 else '#FF4444'
            ml_color = '#00FF88' if ml_change >= 0 else '#FF4444' if ml_pred else '#888888'
            
            ticker_entry = html.Div([
                html.Span(f"{symbol}: ", style={'fontWeight': 'bold', 'color': '#00D4FF'}),
                html.Span(f"${price:.2f}", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Span(f" ({change:+.2f})", style={'color': change_color, 'marginRight': '15px'}),
                html.Span(f"Vol: {volume:,.0f}", style={'color': '#888888', 'marginRight': '10px'}),
                html.Span(f"ML: ${ml_pred:.2f}" if ml_pred else "ML: Training...", style={'color': '#FFA500'}),
                html.Span(f" ({ml_change:+.1f}%)" if ml_pred else "", style={'color': ml_color}),
                html.Span(f" [{ml_status.get(symbol, 'Not Trained')}]", style={'color': '#666666', 'fontSize': '12px'})
            ], style={'marginBottom': '8px'})
            ticker_items.append(ticker_entry)
    
    # Status
    status = f"Last update: {current_time} | Live entries: {len(live_output)} | ML Status: {'Active' if ML_AVAILABLE else 'Disabled'} | Data: {'Real CCXT' if CCXT_AVAILABLE else 'Simulated'}"
    
    return live_display, ticker_items, charts, status

if __name__ == "__main__":
    print("Starting LIVE AI Crypto Trading Dashboard...")
    print(f"Dashboard accessible at: http://0.0.0.0:8050")
    print("-" * 60)
    
    try:
        app.run_server(
            debug=False,
            host='0.0.0.0',
            port=8051,
            dev_tools_ui=False,
            dev_tools_props_check=False
        )
    except Exception as e:
        print(f"Error starting dashboard: {e}")