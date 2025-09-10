import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Análisis de Mejores y Peores Días | BQuant Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced dark theme CSS with animations and glassmorphism
st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Dark theme base with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #0e1117, #1a1f2e, #0e1117, #2d1b69);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* All text uses Inter */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main header with animated gradient */
    .main-header {
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #667eea 75%, #764ba2 100%);
        background-size: 200% 200%;
        animation: gradientFlow 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        font-size: 20px;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Glassmorphism metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.7) 0%, rgba(45, 55, 72, 0.7) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Animated insight boxes */
    .insight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-left: 4px solid;
        border-image: linear-gradient(45deg, #667eea, #764ba2) 1;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        position: relative;
        animation: slideInLeft 0.5s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-left: 4px solid;
        border-image: linear-gradient(45deg, #f56565, #ec4899) 1;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(245, 101, 101, 0.2);
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.15) 0%, rgba(56, 178, 172, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-left: 4px solid;
        border-image: linear-gradient(45deg, #48bb78, #38b2ac) 1;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(72, 187, 120, 0.2);
        animation: fadeInUp 0.5s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced metrics display */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.8) 0%, rgba(45, 55, 72, 0.8) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    /* Glowing tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 12px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 35, 48, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        color: #a0a0a0;
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        border-color: #667eea !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(14, 17, 23, 0.95) 0%, rgba(26, 31, 46, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* Glowing buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Enhanced selectbox */
    .stSelectbox > div > div {
        background: rgba(30, 35, 48, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Enhanced slider */
    .stSlider > div > div {
        background: transparent;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #a0a0a0;
        margin-top: 60px;
        padding: 30px;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.5) 0%, rgba(45, 55, 72, 0.5) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px 20px 0 0;
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .footer a:hover {
        color: #764ba2;
        text-shadow: 0 0 10px rgba(118, 75, 162, 0.5);
    }
    
    /* Number formatting with monospace font */
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
    }
    
    /* Pulse animation for important numbers */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header with enhanced animation
st.markdown('<h1 class="main-header">Tiempo en el Mercado vs Timing del Mercado</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">✨ Analizando el impacto de perder los mejores y peores días de trading</p>', unsafe_allow_html=True)

# Sidebar configuration with better organization
st.sidebar.markdown("## ⚙️ **Configuración**")
st.sidebar.markdown("---")

# Ticker selection
st.sidebar.markdown("### 📊 **Selección de Activo**")
ticker_type = st.sidebar.radio("Tipo de ticker:", ["ETFs de Índices", "Personalizado"], label_visibility="collapsed")

if ticker_type == "ETFs de Índices":
    ticker = st.sidebar.selectbox(
        "Seleccionar ETF/Índice:",
        ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'IVV', '^GSPC', '^DJI', '^IXIC'],
        index=0,
        help="SPY/VOO/IVV = ETFs S&P 500, ^GSPC = Índice S&P 500, ^DJI = Dow Jones, ^IXIC = NASDAQ"
    )
else:
    ticker = st.sidebar.text_input("Ingrese símbolo del ticker:", value="AAPL").upper()

# Date range
st.sidebar.markdown("### 📅 **Período de Análisis**")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Fecha inicial",
        value=pd.to_datetime('2015-01-01'),
        max_value=datetime.now().date()
    )
with col2:
    end_date = st.date_input(
        "Fecha final",
        value=datetime.now().date(),
        min_value=start_date,
        max_value=datetime.now().date()
    )

# Validate date range
if start_date >= end_date:
    st.sidebar.error("⚠️ La fecha final debe ser posterior a la fecha inicial")
    st.stop()

# Days to exclude
st.sidebar.markdown("### 🎯 **Días a Excluir**")
exclude_best = st.sidebar.slider(
    "🟢 Mejores días a perder:",
    min_value=0,
    max_value=50,
    value=10,
    step=1,
    help="Número de mejores días que se perderían al intentar hacer timing del mercado"
)

exclude_worst = st.sidebar.slider(
    "🔴 Peores días a evitar:",
    min_value=0,
    max_value=50,
    value=10,
    step=1,
    help="Número de peores días que se evitarían con timing perfecto (imposible en la práctica)"
)

# Chart settings
st.sidebar.markdown("### 📈 **Configuración de Gráficos**")
scale_type = st.sidebar.radio(
    "Escala del eje Y:",
    ["Lineal", "Logarítmica"],
    index=0,
    help="La escala logarítmica es útil para visualizar cambios porcentuales en períodos largos"
)

# Technical indicators
st.sidebar.markdown("### 📉 **Indicadores Técnicos**")
show_ma = st.sidebar.checkbox("Mostrar Medias Móviles", value=True)
ma_periods = []
if show_ma:
    ma_periods = st.sidebar.multiselect(
        "Períodos MA:",
        [20, 50, 100, 200],
        default=[50, 200],
        help="Medias móviles simples para análisis de tendencia"
    )

# Initial investment
st.sidebar.markdown("### 💰 **Inversión**")
initial_investment = st.sidebar.number_input(
    "Inversión inicial ($):",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Monto inicial para calcular el impacto en dólares"
)

# Load data function with rate limit protection
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start, end):
    """Carga y prepara datos del mercado desde yfinance"""
    import time
    
    # Add minimum delay between yfinance calls
    time.sleep(0.5)
    
    max_retries = 3
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            # Download data using yfinance
            data = yf.download(
                tickers=ticker, 
                start=start, 
                end=end, 
                progress=False, 
                auto_adjust=True,
                threads=False
            )
            
            if data.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
            
            # Handle potential MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Ensure we have Close column
            if 'Close' not in data.columns:
                return None
            
            # Ensure Close is numeric
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
            
            # Remove any rows with NaN in Close
            data = data.dropna(subset=['Close'])
            
            if len(data) < 20:
                return None
            
            # Calculate returns
            data['Daily_Return'] = data['Close'].pct_change()
            data['Cumulative_Return'] = (1 + data['Daily_Return'].fillna(0)).cumprod() - 1
            
            # Calculate moving averages
            for period in [20, 50, 100, 200]:
                if len(data) >= period:
                    data[f'MA{period}'] = data['Close'].rolling(window=period, min_periods=1).mean()
                else:
                    data[f'MA{period}'] = np.nan
            
            # Position relative to MAs
            data['Above_MA50'] = (data['Close'] > data['MA50']).fillna(False)
            data['Above_MA200'] = (data['Close'] > data['MA200']).fillna(False)
            
            # Volatility
            if len(data) >= 20:
                data['Volatility_20'] = data['Daily_Return'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
            else:
                data['Volatility_20'] = data['Daily_Return'].std() * np.sqrt(252)
            
            return data
            
        except Exception as e:
            error_message = str(e)
            
            # Check if it's a rate limit error
            rate_limit_indicators = ['rate limit', '429', 'too many requests', 'exceeded', 'quota']
            is_rate_limited = any(indicator in error_message.lower() for indicator in rate_limit_indicators)
            
            if is_rate_limited:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    with st.spinner(f"⏳ Límite de tasa alcanzado. Esperando {wait_time} segundos..."):
                        time.sleep(wait_time)
                    continue
                else:
                    st.error("❌ Límite de tasa de Yahoo Finance alcanzado.")
                    st.info("Por favor espere 1-2 minutos y vuelva a intentar.")
                    return None
            else:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    st.error(f"Error cargando datos para {ticker}: {error_message}")
                    return None
    
    return None

# Plotly dark theme configuration
plotly_layout = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#ffffff', family='Inter'),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)'
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)'
    ),
    hoverlabel=dict(
        bgcolor='#1a1f2e',
        font_size=14,
        font_family='Inter'
    )
)

# Load data
with st.spinner(f'🔄 Cargando datos de mercado para {ticker}...'):
    data = load_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error(f"❌ No se pudieron cargar datos para **{ticker}**")
    st.info("💡 **Sugerencias:**")
    st.markdown("""
    - Verifique la ortografía del símbolo del ticker
    - Intente con tickers comunes: **SPY**, **QQQ**, **AAPL**, **MSFT**
    - Para índices, use: **^GSPC** (S&P 500), **^DJI** (Dow Jones), **^IXIC** (NASDAQ)
    - Asegúrese de que el rango de fechas sea válido
    """)
    st.stop()

# Verify we have required columns
if 'Close' not in data.columns or 'Daily_Return' not in data.columns:
    st.error("Los datos no tienen las columnas requeridas.")
    st.stop()

# Calculate scenarios
returns = data['Daily_Return'].dropna()

if len(returns) < 100:
    st.warning("⚠️ No hay suficientes puntos de datos para un análisis significativo. Seleccione un rango de fechas más largo.")
    st.stop()

sorted_returns = returns.sort_values(ascending=False)

# Get best and worst days
best_days = sorted_returns.head(max(exclude_best, 20))
worst_days = sorted_returns.tail(max(exclude_worst, 20))

# Create scenarios
scenarios = {
    'Comprar y Mantener': returns,
}

if exclude_best > 0:
    scenarios[f'Perdiendo {exclude_best} Mejores Días'] = returns[~returns.index.isin(sorted_returns.head(exclude_best).index)]

if exclude_worst > 0:
    scenarios[f'Perdiendo {exclude_worst} Peores Días'] = returns[~returns.index.isin(sorted_returns.tail(exclude_worst).index)]

if exclude_best > 0 and exclude_worst > 0:
    scenarios[f'Perdiendo {exclude_best} Mejores y {exclude_worst} Peores'] = returns[~returns.index.isin(
        sorted_returns.head(exclude_best).index.union(sorted_returns.tail(exclude_worst).index)
    )]

# Calculate cumulative returns
cumulative_scenarios = {}
final_returns = {}
dollar_scenarios = {}

for name, rets in scenarios.items():
    if len(rets) > 0:
        cumulative = (1 + rets).cumprod()
        cumulative_scenarios[name] = cumulative
        final_returns[name] = (cumulative.iloc[-1] - 1) * 100
        dollar_scenarios[name] = initial_investment * cumulative
    else:
        cumulative_scenarios[name] = pd.Series([1], index=[pd.Timestamp.now()])
        final_returns[name] = 0
        dollar_scenarios[name] = pd.Series([initial_investment], index=[pd.Timestamp.now()])

# Key metrics
total_days = len(returns)
years = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25, 0.1)

# Display metrics with enhanced styling
st.markdown("### 📊 **Métricas Clave**")
col1, col2, col3, col4 = st.columns(4)

buy_hold_return = final_returns.get('Comprar y Mantener', 0)
buy_hold_value = dollar_scenarios.get('Comprar y Mantener', pd.Series([initial_investment])).iloc[-1]

with col1:
    annualized = ((1 + buy_hold_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    st.metric(
        "🏆 Comprar y Mantener",
        f"${buy_hold_value:,.0f}",
        f"+{buy_hold_return:.1f}% ({annualized:.1f}% anual)"
    )

with col2:
    if f'Perdiendo {exclude_best} Mejores Días' in final_returns and exclude_best > 0:
        miss_best_return = final_returns[f'Perdiendo {exclude_best} Mejores Días']
        miss_best_value = dollar_scenarios[f'Perdiendo {exclude_best} Mejores Días'].iloc[-1]
        diff = miss_best_return - buy_hold_return
        st.metric(
            f"😔 Sin {exclude_best} Mejores",
            f"${miss_best_value:,.0f}",
            f"{diff:.1f}%",
            delta_color="inverse"
        )
    else:
        st.metric(
            "Sin Mejores Días",
            "N/A",
            "Configure días a excluir"
        )

with col3:
    if f'Perdiendo {exclude_worst} Peores Días' in final_returns and exclude_worst > 0:
        miss_worst_return = final_returns[f'Perdiendo {exclude_worst} Peores Días']
        miss_worst_value = dollar_scenarios[f'Perdiendo {exclude_worst} Peores Días'].iloc[-1]
        diff = miss_worst_return - buy_hold_return
        st.metric(
            f"🎯 Sin {exclude_worst} Peores",
            f"${miss_worst_value:,.0f}",
            f"+{diff:.1f}%",
            delta_color="normal"
        )
    else:
        st.metric(
            "Sin Peores Días",
            "N/A",
            "Configure días a excluir"
        )

with col4:
    st.metric(
        "📅 Días de Trading",
        f"{total_days:,}",
        f"{years:.1f} años"
    )

# Create tabs with icons
tab1, tab2, tab3 = st.tabs([
    "📈 Retornos Acumulados",
    "📊 Línea de Tiempo", 
    "📉 Análisis Técnico"
])

with tab1:
    st.subheader("📈 Comparación de Retornos Acumulados")
    
    # Create main chart
    fig = go.Figure()
    
    colors = {
        'Comprar y Mantener': '#48bb78',
        f'Perdiendo {exclude_best} Mejores Días': '#f56565',
        f'Perdiendo {exclude_worst} Peores Días': '#38b2ac',
        f'Perdiendo {exclude_best} Mejores y {exclude_worst} Peores': '#9f7aea'
    }
    
    for name, cumulative in cumulative_scenarios.items():
        color = colors.get(name, '#667eea')
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=(cumulative - 1) * 100,
            mode='lines',
            name=name,
            line=dict(color=color, width=3 if 'Comprar y Mantener' in name else 2.5),
            hovertemplate='<b>%{fullData.name}</b><br>Fecha: %{x}<br>Retorno: %{y:.2f}%<extra></extra>'
        ))
    
    # Add moving averages if selected
    if show_ma and ma_periods:
        for period in ma_periods:
            if f'MA{period}' in data.columns:
                ma_data = data[f'MA{period}'].dropna()
                if len(ma_data) > 0:
                    ma_returns = ma_data.pct_change().fillna(0)
                    ma_cumulative = (1 + ma_returns).cumprod()
                    
                    fig.add_trace(go.Scatter(
                        x=ma_cumulative.index,
                        y=(ma_cumulative - 1) * 100,
                        mode='lines',
                        name=f'MA{period}',
                        line=dict(dash='dot', width=1, color='rgba(255,255,255,0.3)'),
                        hovertemplate=f'<b>MA{period}</b><br>Fecha: %{{x}}<br>Retorno: %{{y:.2f}}%<extra></extra>'
                    ))
    
    # Update layout with proper log scale
    y_axis_config = dict(
        title='Retorno Acumulado (%)',
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)'
    )
    
    if scale_type == 'Logarítmica':
        # For log scale, shift values to be positive
        for trace in fig.data:
            trace.y = [y + 100 for y in trace.y]
        y_axis_config['type'] = 'log'
        y_axis_config['title'] = 'Retorno Acumulado (%, escala log)'
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Inter'),
        xaxis=dict(
            title='Fecha',
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=y_axis_config,
        hoverlabel=dict(
            bgcolor='#1a1f2e',
            font_size=14,
            font_family='Inter'
        ),
        title=dict(
            text=f'<b>{ticker}</b> - Impacto de Perder los Mejores/Peores Días',
            font=dict(size=24, color='#ffffff'),
            x=0.5,
            xanchor='center'
        ),
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.2)',
            borderwidth=1,
            font=dict(size=12)
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.subheader("📊 Resumen de Estrategias")
    
    summary_data = []
    for strategy_name in final_returns.keys():
        if strategy_name in dollar_scenarios:
            final_ret = final_returns[strategy_name]
            final_val = dollar_scenarios[strategy_name].iloc[-1]
            annual_ret = ((1 + final_ret/100) ** (1/years) - 1) * 100 if years > 0 else 0
            
            summary_data.append({
                'Estrategia': strategy_name,
                'Retorno Final (%)': final_ret,
                'Valor Final ($)': final_val,
                'Anualizado (%)': annual_ret
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.round(2), use_container_width=True, height=180)

with tab2:
    st.subheader("📊 Distribución de los Mejores y Peores Días en el Tiempo")
    
    # Timeline scatter plot
    fig_dist = go.Figure()
    
    # Add best days
    fig_dist.add_trace(go.Scatter(
        x=best_days[:exclude_best].index if exclude_best > 0 else best_days[:20].index,
        y=(best_days[:exclude_best] if exclude_best > 0 else best_days[:20]).values * 100,
        mode='markers+text',
        name='Mejores Días',
        marker=dict(
            color='#48bb78',
            size=14,
            symbol='triangle-up',
            line=dict(color='#ffffff', width=1)
        ),
        text=[f"#{i+1}" for i in range(min(exclude_best if exclude_best > 0 else 20, len(best_days)))],
        textposition="top center",
        textfont=dict(size=9, color='#48bb78'),
        hovertemplate='<b>Mejor Día #%{text}</b><br>Fecha: %{x}<br>Retorno: %{y:.2f}%<extra></extra>'
    ))
    
    # Add worst days
    fig_dist.add_trace(go.Scatter(
        x=worst_days[:exclude_worst].index if exclude_worst > 0 else worst_days[:20].index,
        y=(worst_days[:exclude_worst] if exclude_worst > 0 else worst_days[:20]).values * 100,
        mode='markers+text',
        name='Peores Días',
        marker=dict(
            color='#f56565',
            size=14,
            symbol='triangle-down',
            line=dict(color='#ffffff', width=1)
        ),
        text=[f"#{i+1}" for i in range(min(exclude_worst if exclude_worst > 0 else 20, len(worst_days)))],
        textposition="bottom center",
        textfont=dict(size=9, color='#f56565'),
        hovertemplate='<b>Peor Día #%{text}</b><br>Fecha: %{x}<br>Retorno: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line
    fig_dist.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    
    # Add crisis periods
    crisis_periods = [
        ('2000-03-01', '2002-10-01', 'Burbuja puntocom'),
        ('2007-10-01', '2009-03-01', 'Crisis 2008'),
        ('2020-02-01', '2020-04-01', 'COVID-19'),
        ('2022-01-01', '2022-06-01', 'Inflación/Guerra')
    ]
    
    for start, end, name in crisis_periods:
        if pd.to_datetime(start) >= pd.to_datetime(start_date) and pd.to_datetime(end) <= pd.to_datetime(end_date):
            fig_dist.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(255,255,255,0.03)",
                layer="below",
                line_width=0,
                annotation_text=name,
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="rgba(255,255,255,0.5)"
            )
    
    fig_dist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Inter'),
        xaxis=dict(
            title='Fecha',
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Retorno Diario (%)',
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        title=dict(
            text='<b>Distribución de Días Extremos</b>',
            font=dict(size=20, color='#ffffff'),
            x=0.5,
            xanchor='center'
        ),
        height=450,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.2)',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='#1a1f2e',
            font_size=14,
            font_family='Inter'
        )
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Distribution by year
    col1, col2 = st.columns(2)
    
    with col1:
        best_to_show = best_days[:exclude_best] if exclude_best > 0 else best_days[:20]
        if len(best_to_show) > 0:
            best_years = pd.Series(best_to_show.index.year).value_counts().sort_index()
            fig_best = px.bar(
                x=best_years.index,
                y=best_years.values,
                labels={'x': 'Año', 'y': 'Cantidad'},
                title='🟢 Mejores Días por Año'
            )
            fig_best.update_traces(marker_color='#48bb78')
            fig_best.update_layout(**plotly_layout, height=300)
            st.plotly_chart(fig_best, use_container_width=True)
    
    with col2:
        worst_to_show = worst_days[:exclude_worst] if exclude_worst > 0 else worst_days[:20]
        if len(worst_to_show) > 0:
            worst_years = pd.Series(worst_to_show.index.year).value_counts().sort_index()
            fig_worst = px.bar(
                x=worst_years.index,
                y=worst_years.values,
                labels={'x': 'Año', 'y': 'Cantidad'},
                title='🔴 Peores Días por Año'
            )
            fig_worst.update_traces(marker_color='#f56565')
            fig_worst.update_layout(**plotly_layout, height=300)
            st.plotly_chart(fig_worst, use_container_width=True)

with tab3:
    st.subheader("📉 Análisis Técnico de Días Extremos")
    
    # Check position relative to moving averages
    best_days_subset = best_days[:exclude_best] if exclude_best > 0 else best_days[:20]
    worst_days_subset = worst_days[:exclude_worst] if exclude_worst > 0 else worst_days[:20]
    
    # MA Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 **Mejores Días vs Medias Móviles**")
        
        try:
            valid_best_days = best_days_subset.index.intersection(data.index)
            if len(valid_best_days) > 0:
                best_above_ma50 = data.loc[valid_best_days, 'Above_MA50'].sum()
                best_above_ma200 = data.loc[valid_best_days, 'Above_MA200'].sum()
                
                st.markdown(f"""
                <div class="insight-box">
                    <p style="color: #ffffff; font-size: 16px;">
                        <b>{best_above_ma50}</b> de {len(valid_best_days)} mejores días ocurrieron <b>sobre MA50</b><br>
                        <b>{best_above_ma200}</b> de {len(valid_best_days)} mejores días ocurrieron <b>sobre MA200</b>
                    </p>
                    <p style="color: #a0a0a0; font-size: 14px; margin-top: 10px;">
                        💡 Los mejores días pueden ocurrir tanto en mercados alcistas como bajistas
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.info("No hay suficientes datos para análisis de MA")
    
    with col2:
        st.markdown("### 🔴 **Peores Días vs Medias Móviles**")
        
        try:
            valid_worst_days = worst_days_subset.index.intersection(data.index)
            if len(valid_worst_days) > 0:
                worst_above_ma50 = data.loc[valid_worst_days, 'Above_MA50'].sum()
                worst_above_ma200 = data.loc[valid_worst_days, 'Above_MA200'].sum()
                
                st.markdown(f"""
                <div class="warning-box">
                    <p style="color: #ffffff; font-size: 16px;">
                        <b>{worst_above_ma50}</b> de {len(valid_worst_days)} peores días ocurrieron <b>sobre MA50</b><br>
                        <b>{worst_above_ma200}</b> de {len(valid_worst_days)} peores días ocurrieron <b>sobre MA200</b>
                    </p>
                    <p style="color: #a0a0a0; font-size: 14px; margin-top: 10px;">
                        ⚠️ Las caídas pueden ocurrir incluso en tendencias alcistas
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.info("No hay suficientes datos para análisis de MA")
    
    # Price chart with extreme days marked
    st.markdown("### 📊 **Gráfico de Precios con Días Extremos**")
    
    fig_price = go.Figure()
    
    # Add price
    fig_price.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Precio',
        line=dict(color='#667eea', width=1.5),
        hovertemplate='Fecha: %{x}<br>Precio: $%{y:.2f}<extra></extra>'
    ))
    
    # Add moving averages
    if show_ma:
        ma_colors = {'MA20': '#ffeb3b', 'MA50': '#00bcd4', 'MA100': '#ff9800', 'MA200': '#e91e63'}
        for period in ma_periods:
            if f'MA{period}' in data.columns:
                ma_data = data[f'MA{period}'].dropna()
                if len(ma_data) > 0:
                    fig_price.add_trace(go.Scatter(
                        x=ma_data.index,
                        y=ma_data.values,
                        mode='lines',
                        name=f'MA{period}',
                        line=dict(width=1, dash='dash', color=ma_colors.get(f'MA{period}', '#ffffff')),
                        opacity=0.7,
                        hovertemplate=f'MA{period}: $%{{y:.2f}}<extra></extra>'
                    ))
    
    # Mark best days
    valid_best = best_days_subset.index.intersection(data.index)
    if len(valid_best) > 0:
        fig_price.add_trace(go.Scatter(
            x=valid_best,
            y=data.loc[valid_best, 'Close'],
            mode='markers',
            name='Mejores Días',
            marker=dict(
                color='#48bb78',
                size=12,
                symbol='triangle-up',
                line=dict(color='#ffffff', width=1)
            ),
            hovertemplate='<b>Mejor Día</b><br>Fecha: %{x}<br>Precio: $%{y:.2f}<br>Retorno: %{text}<extra></extra>',
            text=[f"{best_days_subset[date]*100:.2f}%" for date in valid_best]
        ))
    
    # Mark worst days
    valid_worst = worst_days_subset.index.intersection(data.index)
    if len(valid_worst) > 0:
        fig_price.add_trace(go.Scatter(
            x=valid_worst,
            y=data.loc[valid_worst, 'Close'],
            mode='markers',
            name='Peores Días',
            marker=dict(
                color='#f56565',
                size=12,
                symbol='triangle-down',
                line=dict(color='#ffffff', width=1)
            ),
            hovertemplate='<b>Peor Día</b><br>Fecha: %{x}<br>Precio: $%{y:.2f}<br>Retorno: %{text}<extra></extra>',
            text=[f"{worst_days_subset[date]*100:.2f}%" for date in valid_worst]
        ))
    
    # Update layout with proper log scale for price
    price_axis_config = dict(
        title='Precio ($)',
        gridcolor='rgba(255,255,255,0.05)',
        zerolinecolor='rgba(255,255,255,0.1)'
    )
    
    if scale_type == 'Logarítmica':
        price_axis_config['type'] = 'log'
        price_axis_config['title'] = 'Precio ($, escala log)'
    
    fig_price.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Inter'),
        xaxis=dict(
            title='Fecha',
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=price_axis_config,
        hoverlabel=dict(
            bgcolor='#1a1f2e',
            font_size=14,
            font_family='Inter'
        ),
        title=dict(
            text=f'<b>{ticker}</b> - Precio con Días Extremos',
            font=dict(size=20, color='#ffffff'),
            x=0.5,
            xanchor='center'
        ),
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.2)',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig_price, use_container_width=True)

# Conclusions section
st.markdown("---")

# Calculate impacts
buy_hold_value = dollar_scenarios.get('Comprar y Mantener', pd.Series([initial_investment])).iloc[-1]
cost_of_missing_best = 0
gain_of_missing_worst = 0

if f'Perdiendo {exclude_best} Mejores Días' in dollar_scenarios:
    cost_of_missing_best = buy_hold_value - dollar_scenarios[f'Perdiendo {exclude_best} Mejores Días'].iloc[-1]

if f'Perdiendo {exclude_worst} Peores Días' in dollar_scenarios:
    gain_of_missing_worst = dollar_scenarios[f'Perdiendo {exclude_worst} Peores Días'].iloc[-1] - buy_hold_value

# Show conclusions if any days are excluded
if exclude_best > 0 or exclude_worst > 0:
    st.markdown(f"""
    <div class="success-box pulse" style="margin-top: 30px;">
        <h2 style="text-align: center; color: #48bb78; font-size: 32px;">🎯 Conclusiones Clave</h2>
    """, unsafe_allow_html=True)
    
    if exclude_best > 0:
        st.markdown(f"""
        <p style="text-align: center; color: #ffffff; font-size: 20px; margin: 15px 0;">
            Perder los <b>{exclude_best} mejores días</b> te habría costado 
            <b style="color: #f56565; font-size: 24px;">${cost_of_missing_best:,.0f}</b>
        </p>
        """, unsafe_allow_html=True)
    
    if exclude_worst > 0:
        st.markdown(f"""
        <p style="text-align: center; color: #ffffff; font-size: 20px; margin: 15px 0;">
            Evitar los <b>{exclude_worst} peores días</b> te habría dado 
            <b style="color: #48bb78; font-size: 24px;">+${gain_of_missing_worst:,.0f}</b> extra
        </p>
        """, unsafe_allow_html=True)
    
    if exclude_best > 0 and exclude_worst > 0:
        combined_value = dollar_scenarios.get(f'Perdiendo {exclude_best} Mejores y {exclude_worst} Peores', pd.Series([initial_investment])).iloc[-1]
        combined_diff = combined_value - buy_hold_value
        if combined_diff > 0:
            st.markdown(f"""
            <p style="text-align: center; color: #ffffff; font-size: 20px; margin: 15px 0;">
                Perder ambos (mejores y peores): 
                <b style="color: #9f7aea; font-size: 24px;">+${combined_diff:,.0f}</b>
            </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <p style="text-align: center; color: #ffffff; font-size: 20px; margin: 15px 0;">
                Perder ambos (mejores y peores): 
                <b style="color: #f56565; font-size: 24px;">${combined_diff:,.0f}</b>
            </p>
            """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 20px; margin-top: 30px;">
            <p style="text-align: center; color: #ffffff; font-size: 18px; margin: 0;">
                <b>⚡ La Verdad Incómoda:</b>
            </p>
            <p style="text-align: center; color: #a0a0a0; font-size: 16px; margin-top: 10px;">
                El timing perfecto es <b>imposible</b>. Los mejores y peores días ocurren cerca unos de otros.<br>
                <span style="color: #48bb78; font-size: 18px; font-weight: 600;">
                    El tiempo en el mercado supera al timing del mercado.
                </span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="success-box" style="margin-top: 30px;">
        <h2 style="text-align: center; color: #48bb78;">Conclusiones Clave</h2>
        <p style="text-align: center; color: #ffffff; font-size: 18px;">
            Comprar y mantener retornó <b>{buy_hold_return:.1f}%</b> en {years:.1f} años
        </p>
        <p style="text-align: center; color: #a0a0a0;">
            El tiempo en el mercado supera al timing del mercado.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer with enhanced styling
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-size: 18px;">
        Hecho con 📈 por <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a> | 
        <a href="https://bquantfinance.com" target="_blank">bquantfinance.com</a>
    </p>
</div>
""", unsafe_allow_html=True)
