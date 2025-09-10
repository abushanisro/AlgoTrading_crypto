# AI-Powered Crypto Trading Dashboard
![Image alt](https://github.com/abushanisro/AlgoTrading_crypto/blob/main/Crytoalgo.png?raw=true)

A sophisticated, high-frequency cryptocurrency trading dashboard with real-time ML predictions, technical analysis and automated trading signals.

![Dashboard Preview](https://img.shields.io/badge/Status-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

### **Real-Time Trading Dashboard**
- **High-frequency updates** every 3 seconds
- **Live cryptocurrency data** from Binance via CCXT
- **Interactive candlestick charts** with technical indicators
- **Web-based interface** with modern dark theme

### **AI/ML Powered Predictions**
- **Scikit-learn Random Forest** models for price prediction
- **Automatic model training** on historical data
- **Confidence scoring** for prediction reliability
- **Multi-symbol support** (BTC, ETH, BNB, ADA, SOL)

### **Advanced Technical Analysis**
- **RSI (Relative Strength Index)** - Momentum oscillator
- **MACD (Moving Average Convergence Divergence)** - Trend following
- **Bollinger Bands** - Volatility indicator
- **Moving Averages** (MA20, MA50) - Trend identification
- **Volume analysis** with moving averages

### **Automated Trading Signals**
- **BUY/SELL signal generation** based on technical indicators
- **Signal reasoning** with detailed explanations
- **Real-time alerts** displayed on dashboard
- **Historical signal tracking** in trading journal

### **Live Web Interface**
- **Real-time trading output** - Live scrolling feed of all activity
- **Market data ticker** - Current prices, changes, and volumes
- **ML status monitoring** - Training progress and prediction status
- **Responsive design** - Works on desktop and mobile

## Technical Stack

- **Backend**: Python 3.8+
- **Web Framework**: Dash (Plotly)
- **Data Sources**: CCXT (Binance API)
- **ML/AI**: Scikit-learn (Random Forest)
- **Technical Analysis**: Custom implementations
- **Charts**: Plotly with real-time updates
- **Data Processing**: Pandas, NumPy

## Installation

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package installer)
```

### Clone Repository
```bash
git clone https://github.com/abushanisro/Crypto_algotade.git
cd Crypto_algotade
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the project root:
```bash
# Optional: Add your API keys for enhanced data access
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_here
```

## Quick Start

### 1. Run the Main Dashboard
```bash
python ai_dashboard_working.py
```

### 2. Run the Live Web Dashboard (Recommended)
```bash
python ai_dashboard_live_web.py
```

### 3. Access the Dashboard
Open your browser and navigate to:
- **Main Dashboard**: http://localhost:8050
- **Live Web Dashboard**: http://localhost:8051

## Dashboard Sections

### **Live Trading Output**
Real-time feed showing:
```
[14:25:12] UPDATE #1 - Processing 5 symbols...
BTC/USDT: UP → $65,890.25 (+0.70%) - 87.3% confidence
TRADING SIGNAL: BTC/USDT BUY at $65,432.50 - RSI oversold
```

### **Real-Time Market Data**
```
Live Update: 14:25:15
BTC/USDT: $65,432.50 (+123.45) Vol: 1,234,567 ML: $65,890.25 (+0.70%) [Trained & Ready]
ETH/USDT: $3,245.80 (-12.30) Vol: 987,654 ML: Training... [Training...]
```

### **Interactive Charts**
- Candlestick price charts
- Technical indicator overlays
- BUY/SELL signal annotations
- Zoom and pan functionality

### **AI Commentary & Strategy**
- Market analysis based on technical indicators
- Strategy recommendations
- Risk assessment
- Entry/exit suggestions

## Configuration

### Symbols
Edit the `SYMBOLS` list in the main files:
```python
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
```

### Update Frequency
Modify the update interval:
```python
UPDATE_INTERVAL = 3000   # 3 seconds (in milliseconds)
```

### ML Model Parameters
Adjust Random Forest parameters:
```python
RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Maximum tree depth
    random_state=42,     # For reproducibility
    n_jobs=-1           # Use all CPU cores
)
```

## Requirements

### Core Dependencies
```
dash>=2.11.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
ccxt>=4.0.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
```

### Optional Dependencies
```
pyttsx3>=2.90    # For voice alerts (if needed)
requests>=2.31.0 # For additional API calls
```

## Key Components

### 1. **Data Fetching** (`fetch_data`)
- Connects to Binance via CCXT
- Fetches 1-minute OHLCV data
- Falls back to simulated data if API unavailable

### 2. **Technical Analysis** (`calculate_technical_indicators`)
- RSI calculation with 14-period
- MACD with 12/26/9 parameters
- Bollinger Bands (20-period, 2 std dev)
- Multiple moving averages

### 3. **ML Prediction** (`CryptoMLPredictor`)
- Feature engineering from price/volume/indicators
- Random Forest training on historical data
- Real-time price predictions with confidence

### 4. **Signal Generation** (`generate_trading_signals`)
- Multi-factor analysis scoring system
- BUY/SELL threshold logic
- Signal reasoning and explanations

### 5. **Web Interface** (Dash app)
- Real-time updates every 3 seconds
- Interactive charts and displays
- Live data streaming to browser

## How It Works

1. **Data Collection**: Fetches real-time crypto data from Binance
2. **Technical Analysis**: Calculates RSI, MACD, Bollinger Bands, etc.
3. **ML Training**: Trains Random Forest models on historical patterns
4. **Prediction**: Generates price predictions with confidence scores
5. **Signal Generation**: Creates BUY/SELL signals based on multiple factors
6. **Visualization**: Displays everything in real-time web dashboard
7. **Monitoring**: Continuously updates every 3 seconds

## Disclaimer

**This software is for educational and research purposes only.**

- **Not financial advice**: Do not use for actual trading without proper research
- **No guarantees**: Past performance doesn't predict future results
- **Use at own risk**: Cryptocurrency trading involves significant risk
- **Test thoroughly**: Always test with paper trading first

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CCXT** - Cryptocurrency trading library
- **Plotly/Dash** - Interactive web applications
- **Scikit-learn** - Machine learning library
- **Binance** - Cryptocurrency exchange API
- **Open Source Community** - For inspiration and tools

## Support

If you find this project helpful, please:
- Star the repository
- Report bugs via GitHub Issues
- Suggest features via GitHub Issues
- Submit Pull Requests

---

**Built with ❤ by the AI Trading Community**

*Happy Trading!*# crypto_algoTrade

