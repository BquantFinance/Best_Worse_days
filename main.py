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
    page_title="Best & Worst Days Analysis | BQuant Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme custom CSS
st.markdown("""
    <style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .sub-header {
        font-size: 18px;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
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
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Time in Market vs Timing the Market</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyzing the impact of missing the best and worst trading days</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Ticker selection
ticker_type = st.sidebar.radio("Select ticker type:", ["Index ETFs", "Custom", "S&P 500 Analysis"])

if ticker_type == "Index ETFs":
    ticker = st.sidebar.selectbox(
        "Select ETF:",
        ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'IVV'],
        index=0
    )
elif ticker_type == "Custom":
    ticker = st.sidebar.text_input("Enter ticker symbol:", value="AAPL").upper()
else:
    ticker = '^GSPC'
    st.sidebar.info("Analyzing S&P 500 Index")

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start date",
        value=pd.to_datetime('2010-01-01'),
        max_value=datetime.now().date()
    )
with col2:
    end_date = st.date_input(
        "End date",
        value=datetime.now().date(),
        min_value=start_date
    )

# Days to exclude
st.sidebar.markdown("### Days to Exclude")
exclude_best = st.sidebar.slider(
    "Best days to miss:",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)

exclude_worst = st.sidebar.slider(
    "Worst days to miss:",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)

# Chart settings
st.sidebar.markdown("### Chart Settings")
scale_type = st.sidebar.radio(
    "Y-axis scale:",
    ["Linear", "Logarithmic"],
    index=0
)

# Technical indicators
st.sidebar.markdown("### Technical Analysis")
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
if show_ma:
    ma_periods = st.sidebar.multiselect(
        "MA Periods:",
        [20, 50, 100, 200],
        default=[50, 200]
    )

# Initial investment
initial_investment = st.sidebar.number_input(
    "Initial investment ($):",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

# Load data function
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, multi_level_index=False)
        if data.empty:
            return None
        
        # Calculate returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
        
        # Calculate moving averages
        for period in [20, 50, 100, 200]:
            data[f'MA{period}'] = data['Close'].rolling(window=period).mean()
        
        # Position relative to MAs
        data['Above_MA50'] = data['Close'] > data['MA50']
        data['Above_MA200'] = data['Close'] > data['MA200']
        
        # Volatility
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

# Load data
with st.spinner('Loading market data...'):
    data = load_data(ticker, start_date, end_date)

if data is None:
    st.error("Unable to load data. Please check the ticker and try again.")
    st.stop()

# Calculate scenarios
returns = data['Daily_Return'].dropna()
sorted_returns = returns.sort_values(ascending=False)

# Get best and worst days
best_days = sorted_returns.head(max(exclude_best, 20))
worst_days = sorted_returns.tail(max(exclude_worst, 20))

# Create scenarios
scenarios = {
    'Buy and Hold': returns,
}

if exclude_best > 0:
    scenarios[f'Missing {exclude_best} Best Days'] = returns[~returns.index.isin(sorted_returns.head(exclude_best).index)]

if exclude_worst > 0:
    scenarios[f'Missing {exclude_worst} Worst Days'] = returns[~returns.index.isin(sorted_returns.tail(exclude_worst).index)]

if exclude_best > 0 and exclude_worst > 0:
    scenarios[f'Missing {exclude_best} Best & {exclude_worst} Worst'] = returns[~returns.index.isin(
        sorted_returns.head(exclude_best).index.union(sorted_returns.tail(exclude_worst).index)
    )]

# Calculate cumulative returns
cumulative_scenarios = {}
final_returns = {}
dollar_scenarios = {}

for name, rets in scenarios.items():
    cumulative = (1 + rets).cumprod()
    cumulative_scenarios[name] = cumulative
    final_returns[name] = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0
    dollar_scenarios[name] = initial_investment * cumulative

# Key metrics
total_days = len(returns)
years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25

# Display metrics
col1, col2, col3, col4 = st.columns(4)

buy_hold_return = final_returns['Buy and Hold']
buy_hold_value = dollar_scenarios['Buy and Hold'].iloc[-1]

with col1:
    st.metric(
        "Buy & Hold",
        f"${buy_hold_value:,.0f}",
        f"+{buy_hold_return:.1f}%"
    )

with col2:
    if f'Missing {exclude_best} Best Days' in final_returns:
        miss_best_return = final_returns[f'Missing {exclude_best} Best Days']
        miss_best_value = dollar_scenarios[f'Missing {exclude_best} Best Days'].iloc[-1]
        st.metric(
            f"Miss {exclude_best} Best",
            f"${miss_best_value:,.0f}",
            f"{miss_best_return - buy_hold_return:.1f}%"
        )

with col3:
    if f'Missing {exclude_worst} Worst Days' in final_returns:
        miss_worst_return = final_returns[f'Missing {exclude_worst} Worst Days']
        miss_worst_value = dollar_scenarios[f'Missing {exclude_worst} Worst Days'].iloc[-1]
        st.metric(
            f"Miss {exclude_worst} Worst",
            f"${miss_worst_value:,.0f}",
            f"+{miss_worst_return - buy_hold_return:.1f}%"
        )

with col4:
    st.metric(
        "Trading Days",
        f"{total_days:,}",
        f"{years:.1f} years"
    )

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Cumulative Returns",
    "📊 Distribution Timeline", 
    "📉 Technical Analysis",
    "📅 Pattern Analysis",
    "🏢 Sector Analysis"
])

with tab1:
    st.subheader("Cumulative Returns Comparison")
    
    # Create main chart
    fig = go.Figure()
    
    colors = {
        'Buy and Hold': '#48bb78',
        f'Missing {exclude_best} Best Days': '#f56565',
        f'Missing {exclude_worst} Worst Days': '#38b2ac',
        f'Missing {exclude_best} Best & {exclude_worst} Worst': '#9f7aea'
    }
    
    for name, cumulative in cumulative_scenarios.items():
        color = colors.get(name, '#667eea')
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=(cumulative - 1) * 100,
            mode='lines',
            name=name,
            line=dict(color=color, width=2.5 if 'Buy and Hold' in name else 2),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    # Add moving averages if selected
    if show_ma and ma_periods:
        for period in ma_periods:
            ma_returns = data[f'MA{period}'].pct_change()
            ma_cumulative = (1 + ma_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=ma_cumulative.index,
                y=(ma_cumulative - 1) * 100,
                mode='lines',
                name=f'MA{period}',
                line=dict(dash='dot', width=1, color='rgba(255,255,255,0.3)'),
                hovertemplate=f'<b>MA{period}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>'
            ))
    
    fig.update_layout(
        **plotly_layout,
        title=f'{ticker} - Impact of Missing Best/Worst Days',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        yaxis_type='log' if scale_type == 'Logarithmic' else 'linear',
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 31, 46, 0.8)',
            bordercolor='rgba(102, 126, 234, 0.2)',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    col1, col2 = st.columns(2)
    
    with col1:
        summary_df = pd.DataFrame({
            'Strategy': list(final_returns.keys()),
            'Final Return (%)': list(final_returns.values()),
            'Final Value ($)': [dollar_scenarios[k].iloc[-1] for k in final_returns.keys()]
        })
        summary_df['Annualized (%)'] = summary_df['Final Return (%)'].apply(
            lambda x: ((1 + x/100) ** (1/years) - 1) * 100
        )
        st.dataframe(summary_df.round(2), use_container_width=True)
    
    with col2:
        opportunity_cost = dollar_scenarios['Buy and Hold'].iloc[-1] - dollar_scenarios.get(
            f'Missing {exclude_best} Best Days', 
            dollar_scenarios['Buy and Hold']
        ).iloc[-1]
        
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="color: #f56565; margin-top: 0;">💸 Opportunity Cost</h3>
            <p style="color: #ffffff; font-size: 16px;">
                Missing {exclude_best} best days costs: <b>${opportunity_cost:,.0f}</b>
            </p>
            <p style="color: #a0a0a0;">
                That's {(opportunity_cost/initial_investment)*100:.1f}% of your initial investment!
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Distribution of Best and Worst Days Over Time")
    
    # Timeline scatter plot
    fig_dist = go.Figure()
    
    # Add best days
    fig_dist.add_trace(go.Scatter(
        x=best_days[:exclude_best].index if exclude_best > 0 else best_days[:20].index,
        y=(best_days[:exclude_best] if exclude_best > 0 else best_days[:20]).values * 100,
        mode='markers',
        name='Best Days',
        marker=dict(
            color='#48bb78',
            size=12,
            symbol='triangle-up',
            line=dict(color='#ffffff', width=1)
        ),
        text=[f"#{i+1}" for i in range(min(exclude_best if exclude_best > 0 else 20, len(best_days)))],
        hovertemplate='<b>Best Day %{text}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Add worst days
    fig_dist.add_trace(go.Scatter(
        x=worst_days[:exclude_worst].index if exclude_worst > 0 else worst_days[:20].index,
        y=(worst_days[:exclude_worst] if exclude_worst > 0 else worst_days[:20]).values * 100,
        mode='markers',
        name='Worst Days',
        marker=dict(
            color='#f56565',
            size=12,
            symbol='triangle-down',
            line=dict(color='#ffffff', width=1)
        ),
        text=[f"#{i+1}" for i in range(min(exclude_worst if exclude_worst > 0 else 20, len(worst_days)))],
        hovertemplate='<b>Worst Day %{text}</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line
    fig_dist.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    
    # Add crisis periods
    crisis_periods = [
        ('2000-03-01', '2002-10-01', 'Dot-com'),
        ('2007-10-01', '2009-03-01', '2008 Crisis'),
        ('2020-02-01', '2020-04-01', 'COVID-19'),
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
        **plotly_layout,
        title='Distribution of Extreme Days',
        xaxis_title='Date',
        yaxis_title='Daily Return (%)',
        height=450,
        showlegend=True
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Distribution by year
    col1, col2 = st.columns(2)
    
    with col1:
        best_years = pd.Series(best_days[:exclude_best if exclude_best > 0 else 20].index.year).value_counts().sort_index()
        fig_best = px.bar(
            x=best_years.index,
            y=best_years.values,
            labels={'x': 'Year', 'y': 'Count'},
            title='Best Days by Year'
        )
        fig_best.update_traces(marker_color='#48bb78')
        fig_best.update_layout(**plotly_layout, height=300)
        st.plotly_chart(fig_best, use_container_width=True)
    
    with col2:
        worst_years = pd.Series(worst_days[:exclude_worst if exclude_worst > 0 else 20].index.year).value_counts().sort_index()
        fig_worst = px.bar(
            x=worst_years.index,
            y=worst_years.values,
            labels={'x': 'Year', 'y': 'Count'},
            title='Worst Days by Year'
        )
        fig_worst.update_traces(marker_color='#f56565')
        fig_worst.update_layout(**plotly_layout, height=300)
        st.plotly_chart(fig_worst, use_container_width=True)

with tab3:
    st.subheader("Technical Analysis of Extreme Days")
    
    # Check position relative to moving averages
    best_days_subset = best_days[:exclude_best] if exclude_best > 0 else best_days[:20]
    worst_days_subset = worst_days[:exclude_worst] if exclude_worst > 0 else worst_days[:20]
    
    # MA Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Best Days vs Moving Averages")
        
        best_above_ma50 = data.loc[best_days_subset.index, 'Above_MA50'].sum()
        best_above_ma200 = data.loc[best_days_subset.index, 'Above_MA200'].sum()
        
        ma_data = pd.DataFrame({
            'Position': ['Above MA50', 'Below MA50', 'Above MA200', 'Below MA200'],
            'Count': [
                best_above_ma50,
                len(best_days_subset) - best_above_ma50,
                best_above_ma200,
                len(best_days_subset) - best_above_ma200
            ]
        })
        
        fig_ma_best = px.bar(
            ma_data[:2],
            x='Position',
            y='Count',
            color='Position',
            color_discrete_map={'Above MA50': '#48bb78', 'Below MA50': '#f56565'},
            title='Best Days Position vs MA50'
        )
        fig_ma_best.update_layout(**plotly_layout, showlegend=False, height=300)
        st.plotly_chart(fig_ma_best, use_container_width=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <p style="color: #ffffff;">
                <b>{best_above_ma200}</b> of {len(best_days_subset)} best days occurred above MA200
            </p>
            <p style="color: #a0a0a0; font-size: 14px;">
                Best days can happen in both bull and bear markets
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Worst Days vs Moving Averages")
        
        worst_above_ma50 = data.loc[worst_days_subset.index, 'Above_MA50'].sum()
        worst_above_ma200 = data.loc[worst_days_subset.index, 'Above_MA200'].sum()
        
        ma_data_worst = pd.DataFrame({
            'Position': ['Above MA50', 'Below MA50', 'Above MA200', 'Below MA200'],
            'Count': [
                worst_above_ma50,
                len(worst_days_subset) - worst_above_ma50,
                worst_above_ma200,
                len(worst_days_subset) - worst_above_ma200
            ]
        })
        
        fig_ma_worst = px.bar(
            ma_data_worst[:2],
            x='Position',
            y='Count',
            color='Position',
            color_discrete_map={'Above MA50': '#48bb78', 'Below MA50': '#f56565'},
            title='Worst Days Position vs MA50'
        )
        fig_ma_worst.update_layout(**plotly_layout, showlegend=False, height=300)
        st.plotly_chart(fig_ma_worst, use_container_width=True)
        
        st.markdown(f"""
        <div class="warning-box">
            <p style="color: #ffffff;">
                <b>{worst_above_ma200}</b> of {len(worst_days_subset)} worst days occurred above MA200
            </p>
            <p style="color: #a0a0a0; font-size: 14px;">
                Crashes can happen even in uptrends
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Price chart with extreme days marked
    st.markdown("### Price Chart with Extreme Days")
    
    fig_price = go.Figure()
    
    # Add price
    fig_price.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#667eea', width=1),
        hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))
    
    # Add moving averages
    if show_ma:
        for period in ma_periods:
            fig_price.add_trace(go.Scatter(
                x=data.index,
                y=data[f'MA{period}'],
                mode='lines',
                name=f'MA{period}',
                line=dict(width=1, dash='dash'),
                hovertemplate=f'MA{period}: %{{y:.2f}}<extra></extra>'
            ))
    
    # Mark best days
    fig_price.add_trace(go.Scatter(
        x=best_days_subset.index,
        y=data.loc[best_days_subset.index, 'Close'],
        mode='markers',
        name='Best Days',
        marker=dict(color='#48bb78', size=10, symbol='triangle-up'),
        hovertemplate='Best Day<br>Return: %{text}<extra></extra>',
        text=[f"{r*100:.2f}%" for r in best_days_subset.values]
    ))
    
    # Mark worst days
    fig_price.add_trace(go.Scatter(
        x=worst_days_subset.index,
        y=data.loc[worst_days_subset.index, 'Close'],
        mode='markers',
        name='Worst Days',
        marker=dict(color='#f56565', size=10, symbol='triangle-down'),
        hovertemplate='Worst Day<br>Return: %{text}<extra></extra>',
        text=[f"{r*100:.2f}%" for r in worst_days_subset.values]
    ))
    
    fig_price.update_layout(
        **plotly_layout,
        title=f'{ticker} Price with Extreme Days',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis_type='log' if scale_type == 'Logarithmic' else 'linear',
        height=500
    )
    
    st.plotly_chart(fig_price, use_container_width=True)

with tab4:
    st.subheader("Pattern Analysis")
    
    # Proximity analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Clustering Analysis")
        
        proximity_windows = [5, 10, 20, 30]
        proximity_counts = []
        
        for window in proximity_windows:
            count = 0
            for best_date in best_days_subset.index:
                for worst_date in worst_days_subset.index:
                    if abs((best_date - worst_date).days) <= window:
                        count += 1
                        break
            proximity_counts.append(count)
        
        proximity_df = pd.DataFrame({
            'Window (days)': proximity_windows,
            'Best Days Near Worst': proximity_counts,
            'Percentage': [c/len(best_days_subset)*100 for c in proximity_counts]
        })
        
        fig_prox = px.bar(
            proximity_df,
            x='Window (days)',
            y='Percentage',
            text='Percentage',
            title='% of Best Days Near Worst Days'
        )
        fig_prox.update_traces(
            marker_color='#9f7aea',
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig_prox.update_layout(**plotly_layout, height=350)
        st.plotly_chart(fig_prox, use_container_width=True)
    
    with col2:
        st.markdown("### Volatility Analysis")
        
        # Volatility on extreme days
        avg_vol = data['Volatility_20'].mean() * 100
        best_vol = data.loc[best_days_subset.index, 'Volatility_20'].mean() * 100
        worst_vol = data.loc[worst_days_subset.index, 'Volatility_20'].mean() * 100
        
        vol_comparison = pd.DataFrame({
            'Period': ['Average', 'Best Days', 'Worst Days'],
            'Volatility (%)': [avg_vol, best_vol, worst_vol]
        })
        
        fig_vol = px.bar(
            vol_comparison,
            x='Period',
            y='Volatility (%)',
            color='Period',
            color_discrete_map={'Average': '#667eea', 'Best Days': '#48bb78', 'Worst Days': '#f56565'},
            title='Volatility Comparison'
        )
        fig_vol.update_layout(**plotly_layout, showlegend=False, height=350)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Day of week analysis
    st.markdown("### Day of Week Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        best_dow = best_days_subset.index.day_name().value_counts()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        best_dow = best_dow.reindex(day_order, fill_value=0)
        
        fig_dow_best = px.bar(
            x=day_order,
            y=best_dow.values,
            labels={'x': 'Day', 'y': 'Count'},
            title='Best Days by Day of Week'
        )
        fig_dow_best.update_traces(marker_color='#48bb78')
        fig_dow_best.update_layout(**plotly_layout, height=300)
        st.plotly_chart(fig_dow_best, use_container_width=True)
    
    with col2:
        worst_dow = worst_days_subset.index.day_name().value_counts()
        worst_dow = worst_dow.reindex(day_order, fill_value=0)
        
        fig_dow_worst = px.bar(
            x=day_order,
            y=worst_dow.values,
            labels={'x': 'Day', 'y': 'Count'},
            title='Worst Days by Day of Week'
        )
        fig_dow_worst.update_traces(marker_color='#f56565')
        fig_dow_worst.update_layout(**plotly_layout, height=300)
        st.plotly_chart(fig_dow_worst, use_container_width=True)

with tab5:
    if ticker_type == "S&P 500 Analysis" or ticker == '^GSPC':
        st.subheader("S&P 500 Sector Analysis")
        
        with st.spinner("Loading S&P 500 sector data..."):
            try:
                # Get S&P 500 components
                sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
                sp500 = sp500[['Symbol', 'GICS Sector']]
                
                # Sample of sectors for demonstration
                sectors_to_analyze = sp500['GICS Sector'].unique()[:5]  # Top 5 sectors
                
                st.markdown("### Extreme Days by Sector")
                
                # Create placeholder for sector analysis
                sector_summary = []
                
                for sector in sectors_to_analyze:
                    sector_tickers = sp500[sp500['GICS Sector'] == sector]['Symbol'].tolist()[:10]  # Sample 10 stocks per sector
                    
                    # Download sector data
                    sector_data = yf.download(sector_tickers, start=start_date, end=end_date, progress=False)['Close']
                    
                    if not sector_data.empty:
                        # Calculate average sector returns
                        sector_returns = sector_data.pct_change().mean(axis=1)
                        
                        # Get best and worst days for sector
                        sector_best = sector_returns.nlargest(5)
                        sector_worst = sector_returns.nsmallest(5)
                        
                        sector_summary.append({
                            'Sector': sector,
                            'Avg Best Day': f"{sector_best.mean()*100:.2f}%",
                            'Avg Worst Day': f"{sector_worst.mean()*100:.2f}%",
                            'Volatility': f"{sector_returns.std()*100*np.sqrt(252):.1f}%"
                        })
                
                if sector_summary:
                    sector_df = pd.DataFrame(sector_summary)
                    st.dataframe(sector_df, use_container_width=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                        <h3 style="color: #667eea; margin-top: 0;">📊 Sector Insights</h3>
                        <p style="color: #a0a0a0;">
                            • Technology and Financials often lead extreme moves<br>
                            • Defensive sectors (Utilities, Staples) show lower volatility<br>
                            • Sector rotation occurs during market extremes
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"Unable to load sector data: {str(e)}")
    else:
        st.info("Select 'S&P 500 Analysis' in the sidebar to see sector analysis")

# Summary section
st.markdown("---")
st.markdown(f"""
<div class="success-box" style="margin-top: 30px;">
    <h2 style="text-align: center; color: #48bb78;">Key Takeaways</h2>
    <p style="text-align: center; color: #ffffff; font-size: 18px;">
        Missing just {exclude_best} best days would have cost you <b>${(dollar_scenarios['Buy and Hold'].iloc[-1] - dollar_scenarios.get(f'Missing {exclude_best} Best Days', dollar_scenarios['Buy and Hold']).iloc[-1]):,.0f}</b>
    </p>
    <p style="text-align: center; color: #a0a0a0;">
        Time in the market beats timing the market.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Made with 📈 by <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a> | 
    <a href="https://bquantfinance.com" target="_blank">bquantfinance.com</a></p>
    <p style="font-size: 12px; color: #666;">
    Educational purposes only. Not investment advice.
    </p>
</div>
""", unsafe_allow_html=True)
