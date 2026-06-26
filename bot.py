import os
import time
import logging
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
BINANCE_API_KEY  = os.environ.get("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_SECRET   = os.environ.get("BINANCE_SECRET",  "YOUR_BINANCE_SECRET")
TESTNET          = os.environ.get("TESTNET", "true").lower() == "true"

# Trading config
SYMBOLS        = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME      = Client.KLINE_INTERVAL_15MINUTE
RISK_PERCENT   = 1.0     # Risk 1% per trade
MIN_SCORE      = 3       # Min signals needed to trade
MAX_TRADES     = 3       # Max concurrent trades
SL_PERCENT     = 0.5     # Stop loss %
TP_PERCENT     = 1.0     # Take profit %

# ─── BINANCE CLIENT ───────────────────────────────────────────────
if TESTNET:
    client = Client(BINANCE_API_KEY, BINANCE_SECRET, testnet=True)
    client.API_URL = "https://testnet.binance.vision/api"
else:
    client = Client(BINANCE_API_KEY, BINANCE_SECRET)

# ─── STRATEGY FUNCTIONS ───────────────────────────────────────────

def get_klines(symbol: str, limit: int = 100) -> pd.DataFrame:
    klines = client.get_klines(symbol=symbol, interval=TIMEFRAME, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume',
        'close_time','quote_volume','trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ])
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df

def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast   = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow   = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_support_resistance(df: pd.DataFrame, lookback: int = 30):
    highs  = df['high'].rolling(5, center=True).max()
    lows   = df['low'].rolling(5, center=True).min()
    swing_highs = df['high'][df['high'] == highs].dropna()
    swing_lows  = df['low'][df['low'] == lows].dropna()
    resistance = swing_highs.iloc[-3:].mean() if len(swing_highs) >= 3 else df['high'].max()
    support    = swing_lows.iloc[-3:].mean()  if len(swing_lows)  >= 3 else df['low'].min()
    return support, resistance

def detect_patterns(df: pd.DataFrame) -> int:
    """Returns +1 bullish, -1 bearish, 0 neutral"""
    o1, h1, l1, c1 = df['open'].iloc[-2], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2]
    o2, h2, l2, c2 = df['open'].iloc[-3], df['high'].iloc[-3], df['low'].iloc[-3], df['close'].iloc[-3]
    body1  = abs(c1 - o1)
    range1 = h1 - l1
    body2  = abs(c2 - o2)

    if range1 == 0:
        return 0

    # Bullish Pin Bar (Hammer)
    lower_wick = min(o1, c1) - l1
    upper_wick = h1 - max(o1, c1)
    if lower_wick > range1 * 0.6 and upper_wick < body1 * 0.3 and c1 > o1:
        return 1

    # Bearish Pin Bar (Shooting Star)
    if upper_wick > range1 * 0.6 and lower_wick < body1 * 0.3 and c1 < o1:
        return -1

    # Bullish Engulfing
    if c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2:
        return 1

    # Bearish Engulfing
    if c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2:
        return -1

    # Morning Star
    if c2 < o2 and body1 > body2 * 1.5 and c1 > (o2 + c2) / 2:
        return 1

    # Evening Star
    if c2 > o2 and body1 > body2 * 1.5 and c1 < (o2 + c2) / 2:
        return -1

    return 0

def get_ma_signal(df: pd.DataFrame) -> int:
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200= df['close'].ewm(span=200, adjust=False).mean()
    curr_price = df['close'].iloc[-1]

    fast_cur  = ema20.iloc[-1]
    fast_prev = ema20.iloc[-2]
    slow_cur  = ema50.iloc[-1]
    slow_prev = ema50.iloc[-2]
    trend     = ema200.iloc[-1]

    # Golden cross
    if fast_prev < slow_prev and fast_cur > slow_cur and curr_price > trend:
        return 1
    # Death cross
    if fast_prev > slow_prev and fast_cur < slow_cur and curr_price < trend:
        return -1
    # Trend aligned
    if fast_cur > slow_cur and curr_price > trend:
        return 1
    if fast_cur < slow_cur and curr_price < trend:
        return -1
    return 0

def analyze_symbol(symbol: str) -> dict:
    """Full analysis — returns signal dict"""
    try:
        df = get_klines(symbol, limit=200)
        curr_price = df['close'].iloc[-1]

        # RSI
        rsi     = calc_rsi(df)
        rsi_val = rsi.iloc[-2]
        rsi_prev= rsi.iloc[-3]
        if rsi_prev < 30 and rsi_val > 30:   rsi_sig = 1
        elif rsi_prev > 70 and rsi_val < 70: rsi_sig = -1
        elif rsi_val < 30:                   rsi_sig = 1
        elif rsi_val > 70:                   rsi_sig = -1
        else:                                rsi_sig = 0

        # MACD
        macd_line, sig_line, histogram = calc_macd(df)
        hist_cur  = histogram.iloc[-2]
        hist_prev = histogram.iloc[-3]
        if hist_prev < 0 and hist_cur > 0:   macd_sig = 1
        elif hist_prev > 0 and hist_cur < 0: macd_sig = -1
        elif hist_cur > 0:                   macd_sig = 1
        elif hist_cur < 0:                   macd_sig = -1
        else:                                macd_sig = 0

        # Support / Resistance
        support, resistance = calc_support_resistance(df)
        sr_range = resistance - support
        if sr_range > 0:
            if curr_price < support + sr_range * 0.15:  sr_sig = 1
            elif curr_price > resistance - sr_range*0.15: sr_sig = -1
            elif curr_price > resistance:                sr_sig = 1   # breakout
            elif curr_price < support:                   sr_sig = -1  # breakdown
            else:                                        sr_sig = 0
        else:
            sr_sig = 0

        # Candlestick patterns
        pattern_sig = detect_patterns(df)

        # Moving Averages
        ma_sig = get_ma_signal(df)

        # Score
        buy_score  = sum(1 for s in [rsi_sig, macd_sig, sr_sig, pattern_sig, ma_sig] if s == 1)
        sell_score = sum(1 for s in [rsi_sig, macd_sig, sr_sig, pattern_sig, ma_sig] if s == -1)

        if buy_score >= MIN_SCORE:
            direction = "BUY"
            score = buy_score
        elif sell_score >= MIN_SCORE:
            direction = "SELL"
            score = sell_score
        else:
            direction = "WAIT"
            score = max(buy_score, sell_score)

        return {
            "symbol":     symbol,
            "price":      curr_price,
            "direction":  direction,
            "score":      score,
            "rsi":        round(rsi_val, 2),
            "macd_hist":  round(hist_cur, 6),
            "support":    round(support, 4),
            "resistance": round(resistance, 4),
            "pattern":    pattern_sig,
            "ma":         ma_sig,
            "signals": {
                "RSI":     rsi_sig,
                "MACD":    macd_sig,
                "S/R":     sr_sig,
                "Pattern": pattern_sig,
                "MA":      ma_sig,
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# ─── TRADE EXECUTION ──────────────────────────────────────────────

def get_balance_usdt() -> float:
    try:
        balance = client.get_asset_balance(asset='USDT')
        return float(balance['free'])
    except:
        return 0.0

def get_open_trades() -> list:
    try:
        orders = client.get_open_orders()
        return orders
    except:
        return []

def place_order(symbol: str, side: str, usdt_amount: float) -> dict:
    try:
        price    = float(client.get_symbol_ticker(symbol=symbol)['price'])
        info     = client.get_symbol_info(symbol)
        step_size= float([f for f in info['filters'] if f['filterType']=='LOT_SIZE'][0]['stepSize'])
        quantity = usdt_amount / price
        precision= int(round(-np.log10(step_size)))
        quantity = round(quantity, precision)

        sl_price = round(price * (1 - SL_PERCENT/100), 2) if side == SIDE_BUY else round(price * (1 + SL_PERCENT/100), 2)
        tp_price = round(price * (1 + TP_PERCENT/100), 2) if side == SIDE_BUY else round(price * (1 - TP_PERCENT/100), 2)

        order = client.create_order(
            symbol    = symbol,
            side      = side,
            type      = ORDER_TYPE_MARKET,
            quantity  = quantity
        )
        return {
            "success":  True,
            "order_id": order['orderId'],
            "symbol":   symbol,
            "side":     side,
            "quantity": quantity,
            "price":    price,
            "sl":       sl_price,
            "tp":       tp_price,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── TELEGRAM BOT ─────────────────────────────────────────────────

async def send_signal(app, analysis: dict):
    """Send signal message to Telegram"""
    if analysis['direction'] == "WAIT":
        return

    emoji = "🟢" if analysis['direction'] == "BUY" else "🔴"
    sig   = analysis['signals']

    def fmt(v): return "✅" if v == 1 else ("❌" if v == -1 else "⚪")

    msg = (
        f"{emoji} *{analysis['direction']} SIGNAL* — `{analysis['symbol']}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price: `${analysis['price']:,.4f}`\n"
        f"⭐ Score: `{analysis['score']}/5`\n\n"
        f"📊 *Indicators:*\n"
        f"  RSI ({analysis['rsi']}): {fmt(sig['RSI'])}\n"
        f"  MACD: {fmt(sig['MACD'])}\n"
        f"  Support/Resistance: {fmt(sig['S/R'])}\n"
        f"  Pattern: {fmt(sig['Pattern'])}\n"
        f"  Moving Average: {fmt(sig['MA'])}\n\n"
        f"📈 Support: `${analysis['support']:,.4f}`\n"
        f"📉 Resistance: `${analysis['resistance']:,.4f}`\n"
        f"⏰ Time: `{datetime.now().strftime('%H:%M:%S')}`"
    )

    keyboard = [
        [
            InlineKeyboardButton(f"✅ Execute {analysis['direction']}", callback_data=f"trade_{analysis['direction']}_{analysis['symbol']}"),
            InlineKeyboardButton("❌ Skip", callback_data="skip"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await app.bot.send_message(
        chat_id    = TELEGRAM_CHAT_ID,
        text       = msg,
        parse_mode = "Markdown",
        reply_markup = reply_markup
    )

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 *SmartTrader Bot* — Active!\n\n"
        "Commands:\n"
        "/scan — Scan all symbols now\n"
        "/balance — Check USDT balance\n"
        "/trades — Open trades\n"
        "/status — Bot status\n"
        "/stop — Stop auto scan\n\n"
        "Bot will auto-scan every 15 minutes ✅"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning all symbols...")
    found = 0
    for symbol in SYMBOLS:
        analysis = analyze_symbol(symbol)
        if analysis and analysis['direction'] != "WAIT":
            await send_signal(context.application, analysis)
            found += 1
            await asyncio.sleep(0.5)
    if found == 0:
        await update.message.reply_text("⚪ No signals found. Market is ranging.")

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bal = get_balance_usdt()
    await update.message.reply_text(f"💰 USDT Balance: `${bal:.2f}`", parse_mode="Markdown")

async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    orders = get_open_trades()
    if not orders:
        await update.message.reply_text("📭 No open trades.")
        return
    msg = "📊 *Open Trades:*\n"
    for o in orders[:5]:
        msg += f"• {o['symbol']} {o['side']} qty:{o['origQty']}\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bal = get_balance_usdt()
    msg = (
        f"✅ *Bot Status: RUNNING*\n\n"
        f"💰 Balance: `${bal:.2f}` USDT\n"
        f"📊 Symbols: `{', '.join(SYMBOLS)}`\n"
        f"⏱ Timeframe: `M15`\n"
        f"🎯 Min Score: `{MIN_SCORE}/5`\n"
        f"⚠️ Risk/Trade: `{RISK_PERCENT}%`\n"
        f"🌐 Mode: `{'TESTNET' if TESTNET else 'LIVE'}`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "skip":
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("⏭ Signal skipped.")
        return

    if data.startswith("trade_"):
        _, direction, symbol = data.split("_")
        bal       = get_balance_usdt()
        trade_amt = bal * (RISK_PERCENT / 100) * 10  # use 10x risk amount
        side      = SIDE_BUY if direction == "BUY" else SIDE_SELL

        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(f"⏳ Executing {direction} on {symbol}...")

        result = place_order(symbol, side, trade_amt)

        if result['success']:
            msg = (
                f"✅ *Order Executed!*\n\n"
                f"Symbol: `{result['symbol']}`\n"
                f"Side: `{result['side']}`\n"
                f"Qty: `{result['quantity']}`\n"
                f"Price: `${result['price']:,.4f}`\n"
                f"🛑 SL: `${result['sl']:,.4f}`\n"
                f"🎯 TP: `${result['tp']:,.4f}`\n"
                f"Order ID: `{result['order_id']}`"
            )
        else:
            msg = f"❌ Order Failed!\nError: `{result['error']}`"

        await query.message.reply_text(msg, parse_mode="Markdown")

# ─── AUTO SCAN LOOP ───────────────────────────────────────────────

async def auto_scan(app):
    """Runs every 15 minutes automatically"""
    await asyncio.sleep(10)  # wait for bot to start
    while True:
        logger.info("Auto scanning...")
        for symbol in SYMBOLS:
            try:
                analysis = analyze_symbol(symbol)
                if analysis and analysis['direction'] != "WAIT":
                    await send_signal(app, analysis)
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
        await asyncio.sleep(900)  # 15 minutes

async def post_init(app):
    asyncio.create_task(auto_scan(app))

# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start",   start_command))
    app.add_handler(CommandHandler("scan",    scan_command))
    app.add_handler(CommandHandler("balance", balance_command))
    app.add_handler(CommandHandler("trades",  trades_command))
    app.add_handler(CommandHandler("status",  status_command))
    app.add_handler(CallbackQueryHandler(button_callback))

    logger.info("SmartTrader Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()