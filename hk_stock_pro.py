# 核心優化：導入必要模塊，禁用多餘依賴，節省內存
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed  # 增加重試機制

# ========== 基礎優化配置 ==========
warnings.filterwarnings('ignore')
st.set_page_config(page_title="港股分析穩定版", layout="wide")
# 關鍵優化：禁用matplotlib交互後端，節省內存
plt.switch_backend('Agg')
# 設置中文字體（輕量級，避免加載大字体文件）
plt.rcParams["font.family"] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams["axes.unicode_minus"] = False

# ========== 輕量級依賴檢查（避免啟動卡頓） ==========
try:
    import yfinance as yf
except ImportError:
    st.error("正在安裝必要依賴...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance==0.2.38"])
    import yfinance as yf

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("正在安裝必要依賴...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.4.2"])
    from sklearn.linear_model import LinearRegression

# ========== 頁面UI（簡化設計，減少渲染負載） ==========
st.title("📈 港股分析系統｜輕量穩定版")
st.markdown("### 專為Streamlit Cloud優化，避免卡頓")

# 熱門港股（只保留數據最穩定的標的，減少異常）
hot_stocks = {
    "騰訊控股 (0700)": "0700",
    "美團-W (3690)": "3690",
    "匯豐控股 (0005)": "0005"
}

option = st.selectbox("選擇港股（數據穩定）", list(hot_stocks.keys()))
default_code = hot_stocks[option]
user_code = st.text_input("輸入港股代碼（4-5位數字）", default_code).strip()
predict_days = st.slider("預測天數", 1, 7, 3)  # 縮減預測天數，減少計算量

# ========== 核心工具函數（輕量級） ==========
def is_trading_day(date):
    """輕量級交易日判斷"""
    return date.weekday() not in [5, 6]

def get_trading_dates(start_date, days):
    """輕量級獲取交易日"""
    trading_dates = []
    current_date = start_date
    while len(trading_dates) < days:
        if is_trading_day(current_date):
            trading_dates.append(current_date)
        current_date += timedelta(days=1)
    return trading_dates

def clean_column_names(df):
    """輕量級列名清洗，避免複雜計算"""
    # 處理多級索引
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    
    # 核心列映射（只保留必要列）
    column_mapping = {
        'date': 'Date', 'datetime': 'Date',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'volume': 'Volume', 'vol': 'Volume'
    }
    final_cols = {}
    for col in df.columns:
        for key in column_mapping.keys():
            if key in col:
                final_cols[col] = column_mapping[key]
                break
    df.rename(columns=final_cols, inplace=True)
    return df

# ========== 關鍵優化：帶重試的輕量級數據獲取 ==========
@st.cache_data(ttl=3600)  # 緩存1小時，避免重複請求
@retry(stop=stop_after_attempt(2), wait=wait_fixed(2))  # 失敗重試2次，間隔2秒
def get_hk_stock_data(symbol):
    """
    核心優化點：
    1. 縮減數據範圍（從3年→60天），減少內存佔用
    2. 縮短超時（10秒），避免卡頓
    3. 重試機制，應對網絡波動
    """
    yf_symbol = f"{symbol}.HK"
    st.info(f"🔍 獲取數據：{yf_symbol}（輕量級60天）")
    
    # 關鍵優化：只拉取最近60天數據
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    try:
        # 輕量級下載：只拉取必要字段，關閉進度條
        df = yf.download(
            yf_symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            timeout=10,  # 超時縮短為10秒
            threads=False,  # 禁用多線程，避免資源爭搶
            group_by='ticker'
        )
        
        if df.empty:
            st.error(f"❌ 未獲取到 {yf_symbol} 數據（代碼錯誤/停牌）")
            return None
        
        df.reset_index(inplace=True)
        df = clean_column_names(df)
        
        # 必要列補全（輕量級）
        required_cols = ["Date", "Close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ 核心字段缺失：{missing_cols}")
            return None
        
        # 補全其他字段（用Close填充，減少計算）
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0
        
        # 最終清洗（只保留必要數據）
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        
        # 數據量檢查（最少10條即可運行）
        if len(df) < 10:
            st.warning(f"⚠️ 數據僅{len(df)}條，結果參考性低")
        
        st.success(f"✅ 獲取 {len(df)} 條數據（輕量級）")
        return df
    
    except Exception as e:
        st.error(f"❌ 數據獲取失敗：{str(e)}")
        st.info("💡 解決：刷新頁面/更換騰訊0700測試")
        return None

# ========== 輕量級技術指標計算 ==========
def calculate_indicators(df):
    """縮減計算量，只保留核心指標"""
    if df is None or len(df) < 5:
        return df
    
    df = df.copy()
    try:
        # 只計算核心指標，刪除多餘計算
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        
        # RSI（簡化計算）
        delta = df["Close"].pct_change()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        st.warning(f"⚠️ 指標計算簡化：{str(e)}")
        return df

# ========== 輕量級支撐壓力/預測 ==========
def calculate_support_resistance(df):
    """簡化支撐壓力計算"""
    try:
        support = df["Low"].rolling(window=10, min_periods=1).min().iloc[-1]
        resistance = df["High"].rolling(window=10, min_periods=1).max().iloc[-1]
        return round(support, 2), round(resistance, 2)
    except:
        return round(df["Low"].iloc[-1], 2), round(df["High"].iloc[-1], 2)

def predict_price(df, days):
    """簡化預測，減少計算量"""
    try:
        df["idx"] = np.arange(len(df))
        x = df["idx"].values.reshape(-1, 1)
        y = df["Close"].values
        
        model = LinearRegression()
        model.fit(x, y)
        
        future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
        pred = model.predict(future_idx)
        slope = model.coef_[0]
        
        return pred, slope
    except Exception as e:
        st.warning(f"⚠️ 預測簡化：{str(e)}")
        pred = [df["Close"].iloc[-1]] * days
        return pred, 0

# ========== 主執行邏輯（輕量級渲染） ==========
if st.button("🚀 開始分析（穩定版）", type="primary"):
    # 輸入驗證（簡化）
    if not user_code.isdigit() or len(user_code) not in [4,5]:
        st.error("❌ 代碼需為4-5位數字（如0700）")
    else:
        # 獲取數據（核心優化）
        df = get_hk_stock_data(user_code)
        if df is None:
            st.stop()
        
        # 計算指標
        df = calculate_indicators(df)
        
        # 核心計算
        sup, res = calculate_support_resistance(df)
        pred, slope = predict_price(df, predict_days)
        last_close = df["Close"].iloc[-1]

        # ========== 輕量級展示 ==========
        # 最新數據（只顯示5條，減少表格渲染負載）
        st.subheader("📊 最新5筆數據")
        show_df = df[["Date","Close","MA5","Volume"]].tail(5)
        show_df = show_df.round({"Close":2, "MA5":2, "Volume":0})
        st.dataframe(show_df, use_container_width=True, height=200)  # 固定高度，減少渲染

        # 價格走勢圖（簡化樣式，減少繪圖負載）
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 價格走勢")
            fig, ax = plt.subplots(figsize=(6,3))  # 縮小圖表尺寸
            ax.plot(df["Date"], df["Close"], label="收盤價", linewidth=1)
            ax.plot(df["Date"], df["MA5"], label="MA5", linewidth=0.8)
            ax.set_title(f"{option} ({user_code}.HK)", fontsize=9)
            ax.set_xlabel("日期", fontsize=7)
            ax.set_ylabel("價格 (HK$)", fontsize=7)
            ax.legend(fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
            plt.xticks(rotation=45)
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.subheader("🛡️ 支撐/壓力位")
            st.info(f"📉 支撐位：{sup} HK$")
            st.info(f"📈 壓力位：{res} HK$")
            if last_close < sup:
                st.success(f"當前價 {last_close:.2f}：超賣區間")
            elif last_close > res:
                st.warning(f"當前價 {last_close:.2f}：超買區間")
            else:
                st.info(f"當前價 {last_close:.2f}：區間震盪")

        # 預測結果（簡化展示）
        st.subheader(f"🔮 未來 {predict_days} 天預測")
        trend = "📈 上漲" if slope > 0 else "📉 下跌" if slope < 0 else "📊 平盤"
        st.success(f"趨勢：{trend} | 當前價：{last_close:.2f} HK$")
        
        # 預測表格（簡化）
        last_trading_day = df["Date"].iloc[-1]
        pred_dates = get_trading_dates(last_trading_day + timedelta(days=1), predict_days)
        pred_df = pd.DataFrame({
            "預測日期": [d.strftime("%Y-%m-%d") for d in pred_dates],
            "預測價格": [round(p, 2) for p in pred[:len(pred_dates)]]
        })
        st.dataframe(pred_df, use_container_width=True, height=150)

# ========== 底部提示（簡化） ==========
st.divider()
st.caption("⚠️ 輕量級優化版｜僅供學習｜數據來源：Yahoo Finance")