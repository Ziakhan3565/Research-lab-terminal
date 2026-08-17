import streamlit as st
import pandas as pd
import datetime

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(page_title="Multi-Asset Terminal & Research Lab", layout="wide")

# Dummy list of coins for demonstration (Replace with your actual COINS_LIST if defined elsewhere)
if 'COINS_LIST' not in globals():
    COINS_LIST = ['BTC/USDT', 'SOL/USDT', 'HYPE/USDT', 'TAOUSDT', 'GOLD(XAUT)USDT', 'FARTCOINUSDT']

# Initialize session state logs if not present
if 'trade_history_log' not in st.session_state:
    st.session_state.trade_history_log = [
        {"timestamp": datetime.datetime.now().isoformat(), "symbol": "BTC/USDT", "outcome": "WIN", "direction": "LONG", "score": 0.85},
        {"timestamp": datetime.datetime.now().isoformat(), "symbol": "SOL/USDT", "outcome": "LOSS", "direction": "SHORT", "score": -0.45},
        {"timestamp": datetime.datetime.now().isoformat(), "symbol": "HYPE/USDT", "outcome": "WIN", "direction": "LONG", "score": 0.65},
        {"timestamp": datetime.datetime.now().isoformat(), "symbol": "BTC/USDT", "outcome": "PENDING", "direction": "LONG", "score": 0.32},
    ]

# Custom CSS for styling
st.markdown("""
<style>
.metric-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
}
.metric-label {
    color: #8b949e;
    font-size: 14px;
    font-weight: 600;
}
.metric-value-green {
    font-size: 20px;
    font-weight: 700;
    color: #00e676;
    margin-top: 4px;
}
.metric-value-blue {
    font-size: 20px;
    font-weight: 700;
    color: #58a6ff;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 10-Paper Research Lab & Multi-Asset Terminal")
st.write("Live automated background scanning, order book depth, signal generation, risk monitoring, and performance analytics.")

# ==========================================
# PERFORMANCE & ANALYTICS SECTION + COIN PROFIT/LOSS BREAKDOWN
# ==========================================
st.markdown("---")
st.subheader("📊 Performance, Analytics & Coin-wise Profit/Loss Breakdown")

if st.session_state.trade_history_log:
    df_log = pd.DataFrame(st.session_state.trade_history_log)
    df_log['dt'] = pd.to_datetime(df_log['timestamp'])
    df_log['date'] = df_log['dt'].dt.date
    
    now_dt = datetime.datetime.now()
    today_date = now_dt.date()
    current_year = now_dt.year
    current_week = now_dt.isocalendar()[1]
    current_month = now_dt.month

    total_wins = len(df_log[df_log['outcome'] == 'WIN'])
    total_losses = len(df_log[df_log['outcome'] == 'LOSS'])
    closed_trades = total_wins + total_losses
    overall_win_rate = (total_wins / closed_trades * 100) if closed_trades > 0 else 0.0

    wr1, wr2, wr3, wr4 = st.columns(4)
    with wr1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Overall Win Rate (All Coins)</div><div class="metric-value-green">{overall_win_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with wr2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Wins (W)</div><div style="font-size:20px; font-weight:700; color:#00e676; margin-top:4px;">{total_wins}</div></div>', unsafe_allow_html=True)
    with wr3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Losses (L)</div><div style="font-size:20px; font-weight:700; color:#ff5252; margin-top:4px;">{total_losses}</div></div>', unsafe_allow_html=True)
    with wr4:
        pending_count = len(df_log[df_log['outcome'] == 'PENDING'])
        st.markdown(f'<div class="metric-card"><div class="metric-label">Pending Outcomes</div><div class="metric-value-blue">{pending_count}</div></div>', unsafe_allow_html=True)

    # ==========================================
    # COIN-WISE PERFORMANCE & PROFIT/LOSS BREAKDOWN
    # ==========================================
    st.markdown("### 🏆 Coin-wise Win/Loss & Profit Ranking")
    coin_perf_list = []
    for coin in COINS_LIST:
        coin_df = df_log[df_log['symbol'] == coin]
        c_wins = len(coin_df[coin_df['outcome'] == 'WIN'])
        c_losses = len(coin_df[coin_df['outcome'] == 'LOSS'])
        c_closed = c_wins + c_losses
        c_wr = (c_wins / c_closed * 100) if c_closed > 0 else 0.0
        c_net_pnl = (c_wins * 4) - (c_losses * 2)  # Adjust R:R multiplier as needed (+4 units per win, -2 per loss)
        
        coin_perf_list.append({
            "Symbol": coin,
            "Wins": c_wins,
            "Losses": c_losses,
            "Win Rate": f"{c_wr:.1f}%",
            "Est. PnL ($)": f"${c_net_pnl:+d}"
        })
    
    df_coin_perf = pd.DataFrame(coin_perf_list)
    df_coin_perf['sort_val'] = df_coin_perf['Est. PnL ($)'].str.replace('$', '').str.replace('+', '').astype(int)
    df_coin_perf = df_coin_perf.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
    
    st.dataframe(df_coin_perf, use_container_width=True, hide_index=True, height=220)

    # Time-based breakdown tabs (Daily, Weekly, Monthly)
    df_today = df_log[df_log['date'] == today_date]
    tot_d = len(df_today)
    long_d = len(df_today[df_today['direction'] == 'LONG']) if tot_d > 0 else 0
    short_d = len(df_today[df_today['direction'] == 'SHORT']) if tot_d > 0 else 0
    avg_s_d = df_today['score'].mean() if tot_d > 0 else 0.0

    df_week = df_log[(df_log['dt'].dt.isocalendar().week == current_week) & (df_log['dt'].dt.year == current_year)]
    tot_w = len(df_week)
    long_w = len(df_week[df_week['direction'] == 'LONG']) if tot_w > 0 else 0
    short_w = len(df_week[df_week['direction'] == 'SHORT']) if tot_w > 0 else 0
    avg_s_w = df_week['score'].mean() if tot_w > 0 else 0.0

    df_month = df_log[(df_log['dt'].dt.month == current_month) & (df_log['dt'].dt.year == current_year)]
    tot_m = len(df_month)
    long_m = len(df_month[df_month['direction'] == 'LONG']) if tot_m > 0 else 0
    short_m = len(df_month[df_month['direction'] == 'SHORT']) if tot_m > 0 else 0
    avg_s_m = df_month['score'].mean() if tot_m > 0 else 0.0

    tab_d, tab_w, tab_m = st.tabs(["📅 Daily Overview", "📈 Weekly Overview", "🗓️ Monthly Overview"])

    with tab_d:
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Signals (Today)</div><div class="metric-value-blue">{tot_d}</div></div>', unsafe_allow_html=True)
        with w2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">LONG / SHORT</div><div style="font-size:18px; font-weight:700; color:#00e676; margin-top:4px;">{long_d} / <span style="color:#ff5252;">{short_d}</span></div></div>', unsafe_allow_html=True)
        with w3:
            sc_col = "#00e676" if avg_s_d >= 0 else "#ff5252"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score (Today)</div><div style="font-size:18px; font-weight:700; color:{sc_col}; margin-top:4px;">{avg_s_d:+.3f}</div></div>', unsafe_allow_html=True)
        with w4:
            neu_d = tot_d - (long_d + short_d)
            st.markdown(f'<div class="metric-card"><div class="metric-label">Neutral Signals</div><div style="font-size:18px; font-weight:700; color:#8b949e; margin-top:4px;">{neu_d}</div></div>', unsafe_allow_html=True)

    with tab_w:
        ww1, ww2, ww3, ww4 = st.columns(4)
        with ww1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Signals (This Week)</div><div class="metric-value-blue">{tot_w}</div></div>', unsafe_allow_html=True)
        with ww2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">LONG / SHORT</div><div style="font-size:18px; font-weight:700; color:#00e676; margin-top:4px;">{long_w} / <span style="color:#ff5252;">{short_w}</span></div></div>', unsafe_allow_html=True)
        with ww3:
            sc_col_w = "#00e676" if avg_s_w >= 0 else "#ff5252"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score (This Week)</div><div style="font-size:18px; font-weight:700; color:{sc_col_w}; margin-top:4px;">{avg_s_w:+.3f}</div></div>', unsafe_allow_html=True)
        with ww4:
            neu_w = tot_w - (long_w + short_w)
            st.markdown(f'<div class="metric-card"><div class="metric-label">Neutral Signals</div><div style="font-size:18px; font-weight:700; color:#8b949e; margin-top:4px;">{neu_w}</div></div>', unsafe_allow_html=True)

    with tab_m:
        mm1, mm2, mm3, mm4 = st.columns(4)
        with mm1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Signals (This Month)</div><div class="metric-value-blue">{tot_m}</div></div>', unsafe_allow_html=True)
        with mm2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">LONG / SHORT</div><div style="font-size:18px; font-weight:700; color:#00e676; margin-top:4px;">{long_m} / <span style="color:#ff5252;">{short_m}</span></div></div>', unsafe_allow_html=True)
        with mm3:
            sc_col_m = "#00e676" if avg_s_m >= 0 else "#ff5252"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score (This Month)</div><div style="font-size:18px; font-weight:700; color:{sc_col_m}; margin-top:4px;">{avg_s_m:+.3f}</div></div>', unsafe_allow_html=True)
        with mm4:
            neu_m = tot_m - (long_m + short_m)
            st.markdown(f'<div class="metric-card"><div class="metric-label">Neutral Signals</div><div style="font-size:18px; font-weight:700; color:#8b949e; margin-top:4px;">{neu_m}</div></div>', unsafe_allow_html=True)
