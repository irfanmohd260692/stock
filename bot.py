import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

BOT_TOKEN = "8657963512:AAGlGcSSYGLnqy4eSxgdYSa9apdDhi0K5lg"
CHAT_IDS = [
    "1070509960",
    "1937479700",
    "5034473353"
    # "2037873693"
]


SYMBOL     ="ETHUSDT"
IST        = pytz.timezone("Asia/Kolkata")

last_signal = None  # in-memory; resets on restart

def send_message(text):
    """Send a message to all configured Telegram chat IDs."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in CHAT_IDS:
        try:
            r = requests.post(
                url,
                data={"chat_id": chat_id.strip(), "text": text},
                timeout=10
            )
            r.raise_for_status()
            print(f"[TELEGRAM] Message sent to {chat_id.strip()}")
        except Exception as e:
            print(f"[ERROR] Telegram failed for {chat_id}: {e}")


def calculate_rsi(series, length=14):
    """
    Wilder's RSI using EWM (alpha = 1/length).
    Returns a Series of RSI values (0–100).
    """
    delta    = series.diff()
    gain     = pd.Series(np.where(delta > 0, delta, 0), index=series.index)
    loss     = pd.Series(np.where(delta < 0, -delta, 0), index=series.index)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_candles(symbol=SYMBOL, resolution="30m", lookback_candles=500):
    # Mapping resolution to seconds for dynamic start time
    res_in_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
    delta_seconds = res_in_seconds.get(resolution, 1800)
    
    end = int(time.time())
    start = end - (lookback_candles * delta_seconds)

    try:
        resp = requests.get(
            "https://api.delta.exchange/v2/history/candles",
            params={
                "symbol": symbol, 
                "resolution": resolution, 
                "start": start, 
                "end": end
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("result"):
            print(f"Warning: No data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data["result"])
        
        # Column mapping and type conversion
        df = df.rename(columns={
            "time": "Open_time", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume"
        })

        # Timezone conversion: UTC -> IST -> Naive
        df["Open_time"] = (
            pd.to_datetime(df["Open_time"], unit='s', utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)
        )
        
        # Sort and clean
        df = df.sort_values("Open_time").drop_duplicates().reset_index(drop=True)
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """
    Applies technical indicators to the OHLCV DataFrame.
    Expects columns: 'High', 'Low', 'Close'
    """
    if df.empty or len(df) < 60:
        print("[WARN] Not enough data to calculate 60-period indicators.")
        return df

    # 1. HLC3 (Typical Price)
    df["hlc3"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # 2. Moving Average (SMA 60)
    df["ma"] = df["hlc3"].rolling(window=60).mean()

    # 3. Mean Deviation (for CCI)
    # Optimized: calculates the mean of absolute differences from the mean
    df["mean_dev"] = df["hlc3"].rolling(window=60).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )

    # 4. CCI (Commodity Channel Index) 
    # Formula: (Price - SMA) / (0.015 * Mean Deviation)
    df["CCI_60"] = (df["hlc3"] - df["ma"]) / (0.015 * df["mean_dev"])

    # 5. CCI Smoothing & Difference
    df["CCI_EMA"] = df["CCI_60"].ewm(span=7, adjust=False).mean()
    df["Diff_CCI"] = df["CCI_60"] - df["CCI_EMA"]

    # 6. Trend & Momentum
    df["EMA7"] = df["Close"].ewm(span=7, adjust=False).mean()
    df["RSI"] = calculate_rsi(df["Close"], length=14)

    return df

def generate_signals(df):
    """
    Analyzes the DataFrame to add a 'Signal' column.
    1. Triggers 'Long/Short Trade' on crossovers with a 2-unit spread.
    2. Maintains the signal status as long as price/momentum stay on the right side of EMA.
    """
    df["Signal"] = "No Trade"

    for i in range(1, len(df)):
        # --- DATA FETCHING ---
        curr_close = df.loc[i, "Close"]
        curr_ema7  = df.loc[i, "EMA7"]
        curr_cci   = df.loc[i, "CCI_60"]
        curr_cci_e = df.loc[i, "CCI_EMA"]

        prev_close = df.loc[i-1, "Close"]
        prev_ema7  = df.loc[i-1, "EMA7"]
        prev_cci   = df.loc[i-1, "CCI_60"]
        prev_cci_e = df.loc[i-1, "CCI_EMA"]
        
        # Previous state
        prev_signal = df.loc[i-1, "Signal"]
        
        price_ema_diff = curr_close - curr_ema7

        # --- 2. NEW ENTRY LOGIC (If no active continuation) ---
        # LONG ENTRY
        if (prev_close < prev_ema7 and prev_cci < prev_cci_e) and \
           (curr_close > curr_ema7 and curr_cci > curr_cci_e) and \
           (price_ema_diff >= 2):
            df.loc[i, "Signal"] = "Long Trade"

        # SHORT ENTRY
        elif (prev_close > prev_ema7 and prev_cci > prev_cci_e) and \
             (curr_close < curr_ema7 and curr_cci < curr_cci_e) and \
             (price_ema_diff <= -2):
            df.loc[i, "Signal"] = "Short Trade"
                 
        # --- 1. CONTINUATION LOGIC ---
        # If we were already in a Long, stay in Long as long as conditions hold
        if prev_signal == "Long Trade":
            if curr_close > curr_ema7 and curr_cci > curr_cci_e:
                df.loc[i, "Signal"] = "Long Trade"
                continue # Skip to next candle, no need to check for new entries

        # If we were already in a Short, stay in Short as long as conditions hold
        elif prev_signal == "Short Trade":
            if curr_close < curr_ema7 and curr_cci < curr_cci_e:
                df.loc[i, "Signal"] = "Short Trade"
                continue # Skip to next candle

    return df

def generate_fake_signals(df):
    """
    1. Identifies 'Fake' setups (Momentum vs Price divergence).
    2. After 3 candles, confirms if the trend persisted.
    """
    df["Fake Signal"] = "No Trade"
    
    # We need a longer range to handle lookback + confirmation window
    # i-3 for the fake setup, and checking i vs i-3 for confirmation
    for i in range(6, len(df)):
        # --- 1. DATA COLLECTION ---
        curr_close = df.loc[i, "Close"]
        curr_ema7  = df.loc[i, "EMA7"]
        curr_cci   = df.loc[i, "CCI_60"]
        curr_cci_e = df.loc[i, "CCI_EMA"]
        
        prev_signal = df.loc[i-1, "Fake Signal"]
        price_ema_diff = curr_close - curr_ema7

        # --- 1. CONTINUATION LOGIC ---
        # Maintain existing trades from the main Signal column
        if prev_signal == "Long Trade":
            if curr_close > curr_ema7 and curr_cci > curr_cci_e:
                df.loc[i, "Fake Signal"] = "Long Trade"
                continue 

        elif prev_signal == "Short Trade":
            if curr_close < curr_ema7 and curr_cci < curr_cci_e:
                df.loc[i, "Fake Signal"] = "Short Trade"
                continue

        # --- 2. DETECT INITIAL FAKE SIGNAL (Current Candle i) ---
        # Lookback Window: indices [i-3, i-2, i-1]
        lookback = df.loc[i-3 : i-1]

        long_fake_setup = (
            (lookback["Close"] < lookback["EMA7"]).all() and 
            (lookback["CCI_60"] > lookback["CCI_EMA"]).all()
        )
        
        short_fake_setup = (
            (lookback["Close"] > lookback["EMA7"]).all() and 
            (lookback["CCI_60"] < lookback["CCI_EMA"]).all()
        )

        # Apply the "Fake" label if conditions met
        if (curr_close > curr_ema7 and curr_cci > curr_cci_e) and \
           long_fake_setup and (price_ema_diff >= 2):
            df.loc[i, "Fake Signal"] = "Long Fake Trade"

        elif (curr_close < curr_ema7 and curr_cci < curr_cci_e) and \
             short_fake_setup and (price_ema_diff <= -2):
            df.loc[i, "Fake Signal"] = "Short Fake Trade"

        # --- 3. CONFIRMATION LOGIC (Check 3 candles after a Fake Signal) ---
        # We look back 3 rows to see if THAT row was a Fake Trade
        past_fake_signal = df.loc[i-4, "Fake Signal"]

        if past_fake_signal == "Long Fake Trade":
            # Check if 3 candles later, the trend is still bullish
            if curr_cci > curr_cci_e and curr_close > curr_ema7:
                df.loc[i, "Fake Signal"] = "Long Trade"  # Note: Writes to your Fake Signal column

        elif past_fake_signal == "Short Fake Trade":
            # Check if 3 candles later, the trend is still bearish
            if curr_cci < curr_cci_e and curr_close < curr_ema7:
                df.loc[i, "Fake Signal"] = "Short Trade"

    return df

def final(df):
    df["Final"] = df.apply(
        lambda row: row["Signal"] if row["Fake Signal"] == "No Trade" else row["Fake Signal"],
        axis=1
    )
    return df

def get_telegram_signal(df, symbol):
    """
    Extract latest signal and format Telegram message.
    Returns: (signal, message)
    """

    row = df.iloc[-1]

    open_time = row["Open_time"].strftime("%Y-%m-%d %H:%M")
    close     = row["Close"]
    signal    = row["Final"]
    rsi       = round(row["RSI"], 2) if "RSI" in df.columns else "N/A"

    # =========================
    # 🎨 Emoji + Label
    # =========================
    if signal == "Long Trade":
        emoji = "🟢"
        label = "LONG TRADE"
    elif signal == "Short Trade":
        emoji = "🔴"
        label = "SHORT TRADE"
    elif signal == "Long Fake Trade":
        emoji = "🟡"
        label = "LONG FAKE ⚠️"
    elif signal == "Short Fake Trade":
        emoji = "🟠"
        label = "SHORT FAKE ⚠️"
    else:
        emoji = "⚪"
        label = "NO TRADE"

    # =========================
    # 📝 Message Format
    # =========================
    message = (
        f"{emoji} *{symbol} Signal Alert*\n"
        f"🕐 Time  : {open_time} IST\n"
        f"💰 Close : {close}\n"
        f"📊 Signal: {label}\n"
        f"📈 RSI   : {rsi}"
    )

    return signal, message

def run_signal_check():
    """Fetch data, compute signals, and send Telegram alert on signal change."""
    global last_signal

    print(f"[INFO] Job triggered at: {datetime.now(IST)}")

    # =========================
    # 📥 Fetch Data
    # =========================
    try:
        df = fetch_candles(SYMBOL)
    except Exception as e:
        print(f"[ERROR] API fetch failed: {e}")
        return

    # =========================
    # ⚙️ Compute Indicators + Signals
    # =========================
    df1 = calculate_indicators(df)
    df2 = generate_signals(df1)
    df3 =generate_fake_signals(df2)
    df_final = final(df3)

    # =========================
    # 📡 Get Telegram Message
    # =========================
    signal, msg = get_telegram_signal(df_final, SYMBOL)

    row = df.iloc[-1]
    open_time = row["Open_time"].strftime("%Y-%m-%d %H:%M")
    close     = row["Close"]
    rsi       = round(row["RSI"], 2) if "RSI" in df.columns else "N/A"

    print(f"[INFO] Signal: {signal} | Close: {close} | RSI: {rsi}")

    # =========================
    # 🔔 Send Alert Only on Change
    # =========================
    if signal != last_signal:

        send_message(msg)

        # =========================
        # 💾 Save to CSV (only selected columns)
        # =========================
        log = pd.DataFrame([{
            "Open_time": open_time,
            "Close": close,
            "Signal": signal,
            "RSI": rsi
        }])

        log.to_csv(
            "sff.csv",
            mode='a',
            header=not os.path.exists("signals.csv"),
            index=False
        )

        print("[LOG] Signal saved to signals.csv")

        # Update last signal
        last_signal = signal

    else:
        print(f"[INFO] Signal unchanged ({signal}), no alert sent.")

run_signal_check()

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler = BlockingScheduler(timezone=IST)

scheduler.add_job(
    run_signal_check,
    trigger=CronTrigger(minute="0,30", second="5", timezone=IST),
    misfire_grace_time=60,
    max_instances=1
)

print(f"[INFO] Scheduler started for {SYMBOL}. Fires at :00:05 and :30:05 IST")
# send_message(f"✅ Bot started for {SYMBOL} — running every 30 mins")

try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    print("[INFO] Scheduler stopped.")
