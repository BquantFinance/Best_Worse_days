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
    page_title="Always Invested? | BQuant Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme custom CSS
st.markdown("""
    <style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    /* Custom fonts and headers */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 20px;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        border-left: 4px solid #f56565;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 178, 172, 0.1) 100%);
        border-left: 4px solid #48bb78;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .footer {
        text-align: center;
        color: #a0a0a0;
        margin-top: 60px;
        padding: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 40px 0;
    }
    
    /* Streamlit specific overrides */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 35, 48, 0.5);
        border-radius: 8px;
        color: #a0a0a0;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-color: #667eea !important;
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# Header with branding
st.markdown('<h1 class="main-header">Should You Always Stay Invested?</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A simple demonstration of why the answer is YES</p>', unsafe_allow_html=True)

# Parameters section
with st.expander("⚙️ Choose Your Analysis", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Investable tickers only
        ticker = st.selectbox(
            "📊 Select Investment",
            ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI'],
            index=0,
            help="Choose an index fund to analyze"
        )
        
        ticker_names = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF',
            'IWM': 'Russell 2000 ETF',
            'DIA': 'Dow Jones ETF',
            'VOO': 'Vanguard S&P 500',
            'VTI': 'Total Market ETF'
        }
        selected_name = ticker_names.get(ticker, ticker)
    
    with col2:
        period = st.selectbox(
            "📅 Time Period",
            ['5 Years', '10 Years', '20 Years', 'Max Available'],
            index=1,
            help="How long to analyze"
        )
        
        # Calculate dates based on period
        end_date = datetime.now().date()
        if period == '5 Years':
            start_date = end_date - timedelta(days=5*365)
        elif period == '10 Years':
            start_date = end_date - timedelta(days=10*365)
        elif period == '20 Years':
            start_date = end_date - timedelta(days=20*365)
        else:
            start_date = pd.to_datetime('2000-01-01').date()
    
    with col3:
        exclude_days = st.slider(
            "❌ Days to Miss",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="How many of the best days would you miss by trying to time the market?"
        )
        
        initial_investment = st.number_input(
            "💵 If You Invested",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Starting investment amount"
        )

# Load data
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    """Load and prepare market data"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        
        # Calculate returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        
        # Volatility for context
        data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Plotly dark theme
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

# Load the data
with st.spinner('Loading market data...'):
    data = load_data(ticker, start_date, end_date)

if data is None:
    st.error("Unable to load data. Please try again.")
    st.stop()

returns = data['Daily_Return'].dropna()
sorted_returns = returns.sort_values(ascending=False)

# Calculate scenarios
scenarios = {}
scenarios['Always Invested'] = returns
scenarios[f'Missing {exclude_days} Best Days'] = returns[~returns.index.isin(sorted_returns.head(exclude_days).index)]
scenarios[f'Missing {exclude_days} Worst Days'] = returns[~returns.index.isin(sorted_returns.tail(exclude_days).index)]

# Calculate cumulative returns and dollar values
cumulative_scenarios = {}
final_returns = {}
dollar_scenarios = {}

for name, rets in scenarios.items():
    cumulative = (1 + rets).cumprod()
    cumulative_scenarios[name] = cumulative
    final_returns[name] = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0
    dollar_scenarios[name] = initial_investment * cumulative

# Get best and worst days
best_days = returns.nlargest(exclude_days)
worst_days = returns.nsmallest(exclude_days)

# Calculate key metrics
always_invested_return = final_returns['Always Invested']
missing_best_return = final_returns[f'Missing {exclude_days} Best Days']
missing_worst_return = final_returns[f'Missing {exclude_days} Worst Days']

years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
total_days = len(returns)

always_invested_value = dollar_scenarios['Always Invested'].iloc[-1]
missing_best_value = dollar_scenarios[f'Missing {exclude_days} Best Days'].iloc[-1]
opportunity_cost = always_invested_value - missing_best_value

# Big answer section
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# The Big Answer
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if always_invested_return > missing_best_return:
        st.markdown(f"""
        <div class="success-box" style="text-align: center; padding: 30px;">
            <h1 style="color: #48bb78; font-size: 64px; margin: 0;">YES!</h1>
            <h2 style="color: #ffffff; margin: 20px 0;">Always Stay Invested</h2>
            <p style="color: #a0a0a0; font-size: 18px;">
                Missing just {exclude_days} days would have cost you <b style="color: #f56565;">${opportunity_cost:,.0f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Always Invested</div>
        <div class="metric-value" style="color: #48bb78;">${always_invested_value:,.0f}</div>
        <div class="metric-label" style="color: #48bb78; font-size: 12px;">
            +{always_invested_return:.1f}% total
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Miss {exclude_days} Best Days</div>
        <div class="metric-value" style="color: #f56565;">${missing_best_value:,.0f}</div>
        <div class="metric-label" style="color: #f56565; font-size: 12px;">
            Lost ${opportunity_cost:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    perfect_timing_value = dollar_scenarios[f'Missing {exclude_days} Worst Days'].iloc[-1]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Perfect Timing</div>
        <div class="metric-value" style="color: #9f7aea;">${perfect_timing_value:,.0f}</div>
        <div class="metric-label" style="color: #9f7aea; font-size: 12px;">
            (Impossible to achieve)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    prob = (exclude_days / total_days * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Chance of Missing</div>
        <div class="metric-value" style="color: #ed8936;">{prob:.1f}%</div>
        <div class="metric-label" style="color: #ed8936; font-size: 12px;">
            Just {exclude_days} of {total_days:,} days
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main visualization tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 The Growth Story", 
    "💸 Your Money's Journey",
    "📅 When Magic Happens", 
    "💡 The Simple Truth"
])

with tab1:
    # Growth comparison chart
    fig = go.Figure()
    
    # Always Invested line
    fig.add_trace(go.Scatter(
        x=cumulative_scenarios['Always Invested'].index,
        y=(cumulative_scenarios['Always Invested'] - 1) * 100,
        mode='lines',
        name='Always Invested',
        line=dict(color='#48bb78', width=3),
        fill='tozeroy',
        fillcolor='rgba(72, 187, 120, 0.1)',
        hovertemplate='<b>Always Invested</b><br>Date: %{x}<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Missing best days line
    fig.add_trace(go.Scatter(
        x=cumulative_scenarios[f'Missing {exclude_days} Best Days'].index,
        y=(cumulative_scenarios[f'Missing {exclude_days} Best Days'] - 1) * 100,
        mode='lines',
        name=f'Missing {exclude_days} Best Days',
        line=dict(color='#f56565', width=2, dash='dash'),
        hovertemplate='<b>Missing Best Days</b><br>Date: %{x}<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Perfect timing (for reference)
    fig.add_trace(go.Scatter(
        x=cumulative_scenarios[f'Missing {exclude_days} Worst Days'].index,
        y=(cumulative_scenarios[f'Missing {exclude_days} Worst Days'] - 1) * 100,
        mode='lines',
        name=f'Perfect Timing (Impossible)',
        line=dict(color='#9f7aea', width=1, dash='dot'),
        opacity=0.5,
        hovertemplate='<b>Perfect Timing</b><br>Date: %{x}<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        **plotly_layout,
        title=dict(
            text=f'<b>{selected_name}</b> | Stay Invested vs Try to Time',
            font=dict(size=24, color='#ffffff'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='',
        yaxis_title='Total Return (%)',
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.2)',
            borderwidth=1,
            font=dict(size=14),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simple explanation
    col1, col2 = st.columns(2)
    
    with col1:
        pct_loss = ((always_invested_return - missing_best_return) / always_invested_return) * 100
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="color: #f56565; margin-top: 0;">📉 The Cost of Market Timing</h3>
            <p style="color: #ffffff; font-size: 16px;">
                By trying to time the market and missing just the {exclude_days} best days:
            </p>
            <ul style="color: #a0a0a0; font-size: 14px;">
                <li>You'd lose <b style="color: #f56565;">{pct_loss:.1f}%</b> of your potential gains</li>
                <li>That's <b style="color: #f56565;">${opportunity_cost:,.0f}</b> less in your pocket</li>
                <li>From missing only <b>{prob:.2f}%</b> of trading days!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        annual_return = ((1 + always_invested_return/100) ** (1/years) - 1) * 100
        st.markdown(f"""
        <div class="success-box">
            <h3 style="color: #48bb78; margin-top: 0;">✅ The Power of Patience</h3>
            <p style="color: #ffffff; font-size: 16px;">
                By simply staying invested the entire time:
            </p>
            <ul style="color: #a0a0a0; font-size: 14px;">
                <li>Your ${initial_investment:,} grew to <b style="color: #48bb78;">${always_invested_value:,.0f}</b></li>
                <li>That's <b style="color: #48bb78;">{annual_return:.1f}%</b> per year on average</li>
                <li>Without any stress of timing the market!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # Dollar value journey
    fig_dollar = go.Figure()
    
    # Add initial investment line
    fig_dollar.add_hline(
        y=initial_investment,
        line_dash="dash",
        line_color="rgba(255,255,255,0.2)",
        annotation_text=f"Initial: ${initial_investment:,}",
        annotation_position="left"
    )
    
    # Always invested
    fig_dollar.add_trace(go.Scatter(
        x=dollar_scenarios['Always Invested'].index,
        y=dollar_scenarios['Always Invested'],
        mode='lines',
        name='Always Invested',
        line=dict(color='#48bb78', width=3),
        fill='tonexty',
        fillcolor='rgba(72, 187, 120, 0.1)',
        hovertemplate='<b>Always Invested</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Missing best days
    fig_dollar.add_trace(go.Scatter(
        x=dollar_scenarios[f'Missing {exclude_days} Best Days'].index,
        y=dollar_scenarios[f'Missing {exclude_days} Best Days'],
        mode='lines',
        name=f'Missing {exclude_days} Best Days',
        line=dict(color='#f56565', width=2),
        hovertemplate='<b>Missing Best Days</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add annotations for final values
    fig_dollar.add_annotation(
        x=dollar_scenarios['Always Invested'].index[-1],
        y=always_invested_value,
        text=f"${always_invested_value:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#48bb78",
        font=dict(color="#48bb78", size=14, family="Inter")
    )
    
    fig_dollar.add_annotation(
        x=dollar_scenarios[f'Missing {exclude_days} Best Days'].index[-1],
        y=missing_best_value,
        text=f"${missing_best_value:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f56565",
        font=dict(color="#f56565", size=14, family="Inter")
    )
    
    fig_dollar.update_layout(
        **plotly_layout,
        title=dict(
            text=f'<b>Your ${initial_investment:,} Investment Journey</b>',
            font=dict(size=24, color='#ffffff'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='',
        yaxis_title='Portfolio Value ($)',
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.2)',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_dollar, use_container_width=True)
    
    # Simple comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gain = always_invested_value - initial_investment
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">You Invested</div>
            <div class="metric-value">${initial_investment:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Always Invested Result</div>
            <div class="metric-value" style="color: #48bb78;">${always_invested_value:,.0f}</div>
            <div class="metric-label" style="color: #48bb78; font-size: 12px;">
                Gain: ${gain:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">If You Tried Timing</div>
            <div class="metric-value" style="color: #f56565;">${missing_best_value:,.0f}</div>
            <div class="metric-label" style="color: #f56565; font-size: 12px;">
                Lost: ${opportunity_cost:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.subheader("📅 The Best and Worst Days - When Do They Happen?")
    
    # Timeline of extreme days
    fig_timeline = go.Figure()
    
    # Add best days
    fig_timeline.add_trace(go.Scatter(
        x=best_days.index,
        y=best_days.values * 100,
        mode='markers+text',
        name='Best Days',
        marker=dict(
            color='#48bb78',
            size=15,
            symbol='triangle-up',
            line=dict(color='#ffffff', width=1)
        ),
        text=[f"+{v*100:.1f}%" for v in best_days.values],
        textposition="top center",
        textfont=dict(color='#48bb78', size=10),
        hovertemplate='<b>Best Day</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Add worst days
    fig_timeline.add_trace(go.Scatter(
        x=worst_days.index,
        y=worst_days.values * 100,
        mode='markers+text',
        name='Worst Days',
        marker=dict(
            color='#f56565',
            size=15,
            symbol='triangle-down',
            line=dict(color='#ffffff', width=1)
        ),
        text=[f"{v*100:.1f}%" for v in worst_days.values],
        textposition="bottom center",
        textfont=dict(color='#f56565', size=10),
        hovertemplate='<b>Worst Day</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    
    # Highlight clustering
    for i, best_date in enumerate(best_days.index):
        for worst_date in worst_days.index:
            if abs((best_date - worst_date).days) <= 30:
                fig_timeline.add_shape(
                    type="line",
                    x0=best_date, x1=worst_date,
                    y0=best_days.iloc[i]*100, y1=worst_days[worst_date]*100,
                    line=dict(color="rgba(255,255,255,0.1)", width=1)
                )
                break
    
    fig_timeline.update_layout(
        **plotly_layout,
        title=dict(
            text='<b>Best and Worst Days Often Happen Together!</b>',
            font=dict(size=20, color='#ffffff'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='',
        yaxis_title='Daily Return (%)',
        height=450,
        showlegend=True
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Clustering analysis
    proximity_count = 0
    for best_date in best_days.index:
        for worst_date in worst_days.index:
            if abs((best_date - worst_date).days) <= 30:
                proximity_count += 1
                break
    
    proximity_pct = (proximity_count / len(best_days)) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h3 style="color: #667eea; margin-top: 0;">🎢 Why You Can't Time the Market</h3>
            <p style="color: #ffffff; font-size: 16px;">
                <b>{proximity_pct:.0f}%</b> of the best days happened within 30 days of a worst day!
            </p>
            <p style="color: #a0a0a0; font-size: 14px;">
                • Markets bounce back quickly after crashes<br>
                • If you panic sell, you miss the recovery<br>
                • The best gains come right after the worst losses
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate volatility on extreme days
        best_vol = data.loc[data.index.isin(best_days.index), 'Volatility_20'].mean() * 100
        normal_vol = data['Volatility_20'].mean() * 100
        vol_ratio = best_vol / normal_vol if normal_vol > 0 else 1
        
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="color: #f56565; margin-top: 0;">⚡ Extreme Days Are Unpredictable</h3>
            <p style="color: #ffffff; font-size: 16px;">
                Volatility is <b>{vol_ratio:.1f}x higher</b> on the best days
            </p>
            <p style="color: #a0a0a0; font-size: 14px;">
                • High volatility = high uncertainty<br>
                • Emotions run high, decisions are poor<br>
                • No one can consistently predict these days
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown(f"""
    <div class="success-box" style="text-align: center; padding: 40px;">
        <h1 style="color: #48bb78; margin-top: 0;">The Simple Truth</h1>
        <p style="color: #ffffff; font-size: 20px; margin: 20px 0;">
            <b>"Time in the market beats timing the market"</b>
        </p>
        <p style="color: #a0a0a0; font-size: 16px;">
            It's not about being smart. It's about being patient.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #48bb78; margin-top: 0;">✅ What Works</h4>
            <ul style="color: #a0a0a0; font-size: 14px; list-style: none; padding: 0;">
                <li>📈 Buy and hold</li>
                <li>💵 Regular investing</li>
                <li>😌 Stay calm in crashes</li>
                <li>🎯 Focus on long-term</li>
                <li>🔄 Reinvest dividends</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #f56565; margin-top: 0;">❌ What Doesn't</h4>
            <ul style="color: #a0a0a0; font-size: 14px; list-style: none; padding: 0;">
                <li>📉 Panic selling</li>
                <li>⏰ Market timing</li>
                <li>📰 Following news</li>
                <li>😰 Emotional decisions</li>
                <li>🎰 Day trading</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #9f7aea; margin-top: 0;">📊 Your Results</h4>
            <ul style="color: #a0a0a0; font-size: 14px; list-style: none; padding: 0;">
                <li>📅 {years:.1f} years analyzed</li>
                <li>📈 {total_days:,} trading days</li>
                <li>💰 ${always_invested_value:,.0f} final value</li>
                <li>❌ ${opportunity_cost:,.0f} timing cost</li>
                <li>✅ {(always_invested_value/initial_investment):.1f}x your money</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Final message
    st.markdown(f"""
    <div class="insight-box" style="margin-top: 40px;">
        <h3 style="text-align: center; color: #667eea;">🎯 The Bottom Line</h3>
        <p style="text-align: center; color: #ffffff; font-size: 18px;">
            Over {years:.0f} years, staying invested would have turned your <b>${initial_investment:,}</b> into <b>${always_invested_value:,.0f}</b>.
        </p>
        <p style="text-align: center; color: #ffffff; font-size: 18px;">
            Missing just {exclude_days} days trying to time the market would have cost you <b style="color: #f56565;">${opportunity_cost:,.0f}</b>.
        </p>
        <p style="text-align: center; color: #a0a0a0; font-size: 16px; margin-top: 20px;">
            The best investment strategy? <b>Start now. Stay invested. Be patient.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>Made with 📈 by <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a> | 
    <a href="https://bquantfinance.com" target="_blank">bquantfinance.com</a></p>
    <p style="font-size: 12px; color: #666; margin-top: 10px;">
    Educational purposes only. Not investment advice. Past performance doesn't guarantee future results.
    </p>
</div>
""", unsafe_allow_html=True)
