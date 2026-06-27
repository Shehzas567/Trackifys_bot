import os
import time
import logging
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Trading config
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT",
    "XRP-USDT", "ADA-USDT", "DOGE-USDT", "DOT-USDT",
    "MATIC-USDT", "LINK-USDT", "AVAX-USDT", "UNI-USDT",
    "ATOM-USDT", "LTC-USDT", "TRX-USDT", "APT-USDT",
    "ARB-USDT", "OP-USDT", "NEAR-USDT", "FIL-USDT"
]

MIN_SCORE  = 3
OKX_BASE   = "https://www.okx.com/api/v5"

# ─── OKX DATA FETCH ───────────────────────────────────────────────

def get_candles(symbol: str, bar: str = "15m", limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV candles from OKX (no API key needed for market data)"""
    try:
        url = f"{OKX_BASE}/market/candles"
        params = {"instId": symbol, "bar": bar, "limit": limit}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        if data.get("code") != "0" or not data.get("data"):
            return pd.DataFrame()

        rows = []
        for candle in reversed(data["data"]):
            rows.append({
                "time":   int(candle[0]),
                "open":   float(candle[1]),
                "high":   float(candle[2]),
                "low":    float(candle[3]),
                "close":  float(candle[4]),
                "volume": float(candle[5]),
            })

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        logger.error(f"OKX candle error {symbol}: {e}")
        return pd.DataFrame()

def get_ticker(symbol: str) -> dict:
    """Get current ticker price"""
    try:
        url = f"{OKX_BASE}/market/ticker"
        res = requests.get(url, params={"instId": symbol}, timeout=10)
        data = res.json()
        if data.get("code") == "0" and data.get("data"):
            t = data["data"][0]
            return {
                "price":  float(t["last"]),
                "change": float(t["change24h"]) * 100 if t.get("change24h") else 0,
                "vol24h": float(t["vol24h"]) if t.get("vol24h") else 0,
            }
    except Exception as e:
        logger.error(f"OKX ticker error {symbol}: {e}")
    return {}

# ─── STRATEGY ─────────────────────────────────────────────────────

def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta    = closes.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)

def calc_macd(closes: pd.Series, fast=12, slow=26, signal=9) -> dict:
    if len(closes) < slow:
        return {"hist": 0, "main": 0, "signal": 0}
    ema_fast   = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow   = closes.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line
    return {
        "hist":   round(histogram.iloc[-1], 8),
        "main":   round(macd_line.iloc[-1], 8),
        "signal": round(sig_line.iloc[-1],  8),
    }

def calc_sr(df: pd.DataFrame) -> dict:
    highs      = df["high"].rolling(5, center=True).max()
    lows       = df["low"].rolling(5,  center=True).min()
    swing_highs = df["high"][df["high"] == highs].dropna()
    swing_lows  = df["low"][df["low"]   == lows].dropna()
    resistance  = swing_highs.iloc[-3:].mean() if len(swing_highs) >= 3 else df["high"].max()
    support     = swing_lows.iloc[-3:].mean()  if len(swing_lows)  >= 3 else df["low"].min()
    return {"support": round(support, 6), "resistance": round(resistance, 6)}

def detect_pattern(df: pd.DataFrame) -> int:
    if len(df) < 3:
        return 0
    o1, h1, l1, c1 = df["open"].iloc[-2], df["high"].iloc[-2], df["low"].iloc[-2], df["close"].iloc[-2]
    o2, h2, l2, c2 = df["open"].iloc[-3], df["high"].iloc[-3], df["low"].iloc[-3], df["close"].iloc[-3]
    body1  = abs(c1 - o1)
    range1 = h1 - l1
    if range1 == 0:
        return 0

    # Bullish Pin Bar
    lower_wick = min(o1, c1) - l1
    upper_wick = h1 - max(o1, c1)
    if lower_wick > range1 * 0.6 and c1 > o1:
        return 1
    # Bearish Pin Bar
    if upper_wick > range1 * 0.6 and c1 < o1:
        return -1
    # Bullish Engulfing
    if c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2:
        return 1
    # Bearish Engulfing
    if c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2:
        return -1
    return 0

def calc_ma_signal(closes: pd.Series) -> int:
    if len(closes) < 50:
        return 0
    ma20  = closes.rolling(20).mean().iloc[-1]
    ma50  = closes.rolling(50).mean().iloc[-1]
    ma200 = closes.rolling(min(200, len(closes))).mean().iloc[-1]
    curr  = closes.iloc[-1]
    if curr > ma20 > ma50 and curr > ma200:
        return 1
    if curr < ma20 < ma50 and curr < ma200:
        return -1
    return 0

def analyze(symbol: str) -> dict | None:
    """Full analysis for one symbol"""
    try:
        df = get_candles(symbol, bar="15m", limit=120)
        if df.empty or len(df) < 30:
            return None

        closes = df["close"]
        curr   = closes.iloc[-1]

        # Indicators
        rsi     = calc_rsi(closes)
        macd    = calc_macd(closes)
        sr      = calc_sr(df)
        pattern = detect_pattern(df)
        ma_sig  = calc_ma_signal(closes)

        # Signals
        rsi_sig  = 1 if rsi < 35 else (-1 if rsi > 65 else 0)
        macd_sig = 1 if macd["hist"] > 0 else (-1 if macd["hist"] < 0 else 0)
        sr_range = sr["resistance"] - sr["support"]
        sr_sig   = 0
        if sr_range > 0:
            if curr < sr["support"]    + sr_range * 0.1: sr_sig =  1
            if curr > sr["resistance"] - sr_range * 0.1: sr_sig = -1

        signals = {
            "RSI":     rsi_sig,
            "MACD":    macd_sig,
            "S/R":     sr_sig,
            "Pattern": pattern,
            "MA":      ma_sig,
        }

        buy_score  = sum(1 for v in signals.values() if v ==  1)
        sell_score = sum(1 for v in signals.values() if v == -1)

        direction = "BUY" if buy_score >= MIN_SCORE else "SELL" if sell_score >= MIN_SCORE else "WAIT"
        score     = buy_score if direction == "BUY" else sell_score if direction == "SELL" else max(buy_score, sell_score)

        # Price format
        price_str = f"{curr:.2f}" if curr > 100 else f"{curr:.4f}" if curr > 1 else f"{curr:.6f}"

        return {
            "symbol":     symbol,
            "price":      price_str,
            "direction":  direction,
            "score":      score,
            "rsi":        rsi,
            "macd_hist":  macd["hist"],
            "support":    sr["support"],
            "resistance": sr["resistance"],
            "signals":    signals,
        }

    except Exception as e:
        logger.error(f"Analysis error {symbol}: {e}")
        return None

# ─── TELEGRAM MESSAGE ─────────────────────────────────────────────

async def send_signal(app, result: dict):
    if result["direction"] == "WAIT":
        return

    emoji   = "🟢" if result["direction"] == "BUY" else "🔴"
    sym     = result["symbol"].replace("-", "/")
    signals = result["signals"]

    def fmt(v): return "✅" if v == 1 else ("❌" if v == -1 else "⚪")

    msg = (
        f"{emoji} *{result['direction']} SIGNAL* — `{sym}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price:      `${result['price']}`\n"
        f"⭐ Score:      `{result['score']}/5`\n"
        f"📊 RSI:        `{result['rsi']}`\n\n"
        f"*Indicators:*\n"
        f"  RSI:     {fmt(signals['RSI'])}\n"
        f"  MACD:    {fmt(signals['MACD'])}\n"
        f"  S/R:     {fmt(signals['S/R'])}\n"
        f"  Pattern: {fmt(signals['Pattern'])}\n"
        f"  MA:      {fmt(signals['MA'])}\n\n"
        f"📈 Support:    `${result['support']}`\n"
        f"📉 Resistance: `${result['resistance']}`\n"
        f"⏰ Time: `{datetime.now().strftime('%H:%M:%S')}`"
    )

    keyboard = [[
        InlineKeyboardButton("📊 Detail", callback_data=f"detail_{result['symbol']}"),
        InlineKeyboardButton("❌ Skip",   callback_data="skip"),
    ]]

    await app.bot.send_message(
        chat_id      = TELEGRAM_CHAT_ID,
        text         = msg,
        parse_mode   = "Markdown",
        reply_markup = InlineKeyboardMarkup(keyboard),
    )
    logger.info(f"Signal sent: {result['direction']} {result['symbol']} score={result['score']}")

# ─── COMMANDS ─────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 *CryptoSignal Bot — OKX Edition*\n\n"
        "Commands:\n"
        "/scan    — Scan all 20 coins now\n"
        "/price   — Current prices\n"
        "/status  — Bot status\n"
        "/help    — Show this menu\n\n"
        "✅ Bot is active — scanning every 15 min!"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning 20 coins on OKX...")
    found = 0
    for symbol in SYMBOLS:
        result = analyze(symbol)
        if result and result["direction"] != "WAIT":
            await send_signal(context.application, result)
            found += 1
            await asyncio.sleep(0.5)
    if found == 0:
        await update.message.reply_text("⚪ No strong signals right now. Market is ranging.")
    else:
        await update.message.reply_text(f"✅ Scan done — {found} signal(s) found!")

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Fetching prices...")
    msg = "💰 *Current Prices (OKX):*\n\n"
    for symbol in SYMBOLS[:10]:
        ticker = get_ticker(symbol)
        if ticker:
            sym   = symbol.replace("-USDT", "")
            price = ticker["price"]
            pstr  = f"{price:.2f}" if price > 100 else f"{price:.4f}" if price > 1 else f"{price:.6f}"
            chg   = ticker["change"]
            emoji = "🟢" if chg >= 0 else "🔴"
            msg  += f"{emoji} `{sym}`: ${pstr} ({chg:+.2f}%)\n"
        await asyncio.sleep(0.1)
    await update.message.reply_text(msg, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"✅ *Bot Status: RUNNING*\n\n"
        f"📊 Exchange: `OKX`\n"
        f"🪙 Coins: `{len(SYMBOLS)}`\n"
        f"⏱ Timeframe: `M15`\n"
        f"🎯 Min Score: `{MIN_SCORE}/5`\n"
        f"🔄 Auto scan: `Every 15 min`\n"
        f"⏰ Time: `{datetime.now().strftime('%H:%M:%S')}`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "skip":
        await query.edit_message_reply_markup(reply_markup=None)
    elif query.data.startswith("detail_"):
        symbol = query.data.replace("detail_", "")
        result = analyze(symbol)
        if result:
            sym = symbol.replace("-", "/")
            msg = (
                f"📊 *{sym} Details*\n\n"
                f"Price: `${result['price']}`\n"
                f"RSI: `{result['rsi']}`\n"
                f"Support: `${result['support']}`\n"
                f"Resistance: `${result['resistance']}`\n"
                f"Signal: `{result['direction']}`\n"
                f"Score: `{result['score']}/5`"
            )
            await query.message.reply_text(msg, parse_mode="Markdown")

# ─── AUTO SCAN ────────────────────────────────────────────────────

async def auto_scan(app):
    await asyncio.sleep(15)
    while True:
        logger.info("Auto scan starting...")
        for symbol in SYMBOLS:
            try:
                result = analyze(symbol)
                if result and result["direction"] != "WAIT":
                    await send_signal(app, result)
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Auto scan error {symbol}: {e}")
        logger.info("Auto scan done. Next in 15 min.")
        await asyncio.sleep(900)

async def post_init(app):
    asyncio.create_task(auto_scan(app))

# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set!")
        return
    if not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_CHAT_ID not set!")
        return

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",  start_command))
    app.add_handler(CommandHandler("scan",   scan_command))
    app.add_handler(CommandHandler("price",  price_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help",   start_command))
    app.add_handler(CallbackQueryHandler(button_callback))

    logger.info(f"CryptoSignal Bot starting — {len(SYMBOLS)} coins on OKX")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()