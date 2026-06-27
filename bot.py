import os
import time
import logging
import asyncio
import requests
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
OKX_BASE         = "https://www.okx.com/api/v5"
MIN_SCORE        = 3

SYMBOLS = [
    "BTC-USDT","ETH-USDT","BNB-USDT","SOL-USDT","XRP-USDT",
    "ADA-USDT","DOGE-USDT","DOT-USDT","LINK-USDT","AVAX-USDT",
    "UNI-USDT","ATOM-USDT","LTC-USDT","TRX-USDT","APT-USDT",
    "ARB-USDT","OP-USDT","NEAR-USDT","FIL-USDT","MATIC-USDT"
]

# ─── OKX DATA ────────────────────────────────────────────────────

def get_candles(symbol, bar="15m", limit=100):
    try:
        url = f"{OKX_BASE}/market/candles"
        r = requests.get(url, params={"instId": symbol, "bar": bar, "limit": limit}, timeout=10)
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            return []
        candles = []
        for c in reversed(data["data"]):
            candles.append({
                "open":  float(c[1]),
                "high":  float(c[2]),
                "low":   float(c[3]),
                "close": float(c[4]),
                "vol":   float(c[5]),
            })
        return candles
    except Exception as e:
        logger.error(f"Candle error {symbol}: {e}")
        return []

def get_ticker(symbol):
    try:
        r = requests.get(f"{OKX_BASE}/market/ticker", params={"instId": symbol}, timeout=10)
        d = r.json()
        if d.get("code") == "0" and d.get("data"):
            t = d["data"][0]
            return {"price": float(t["last"]), "change": float(t.get("change24h", 0)) * 100}
    except:
        pass
    return {}

# ─── STRATEGY (pure Python, no pandas) ──────────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    gains  = gains[-period:]
    losses = losses[-period:]
    ag = sum(gains)  / period
    al = sum(losses) / period
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 2)

def calc_ema(values, span):
    if not values:
        return 0
    k = 2 / (span + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema

def calc_macd(closes):
    if len(closes) < 26:
        return 0
    fast = calc_ema(closes, 12)
    slow = calc_ema(closes, 26)
    return fast - slow

def calc_ma(closes, period):
    if len(closes) < period:
        return closes[-1] if closes else 0
    return sum(closes[-period:]) / period

def calc_sr(candles):
    highs = [c["high"] for c in candles[-30:]]
    lows  = [c["low"]  for c in candles[-30:]]
    return {"support": min(lows) * 1.005, "resistance": max(highs) * 0.995}

def detect_pattern(candles):
    if len(candles) < 3:
        return 0
    c1 = candles[-2]
    c2 = candles[-3]
    o1, h1, l1, cl1 = c1["open"], c1["high"], c1["low"], c1["close"]
    o2, h2, l2, cl2 = c2["open"], c2["high"], c2["low"], c2["close"]
    body1  = abs(cl1 - o1)
    range1 = h1 - l1
    if range1 == 0:
        return 0
    lw = min(o1, cl1) - l1
    uw = h1 - max(o1, cl1)
    if lw > range1 * 0.6 and cl1 > o1: return  1  # Hammer
    if uw > range1 * 0.6 and cl1 < o1: return -1  # Shooting Star
    if cl2 < o2 and cl1 > o1 and o1 <= cl2 and cl1 >= o2: return  1  # Bull Engulf
    if cl2 > o2 and cl1 < o1 and o1 >= cl2 and cl1 <= o2: return -1  # Bear Engulf
    return 0

def analyze(symbol):
    try:
        candles = get_candles(symbol, limit=120)
        if len(candles) < 30:
            return None

        closes = [c["close"] for c in candles]
        curr   = closes[-1]

        rsi     = calc_rsi(closes)
        macd_h  = calc_macd(closes)
        sr      = calc_sr(candles)
        pattern = detect_pattern(candles)
        ma20    = calc_ma(closes, 20)
        ma50    = calc_ma(closes, 50)
        ma200   = calc_ma(closes, min(200, len(closes)))

        rsi_sig  = 1 if rsi < 35 else (-1 if rsi > 65 else 0)
        macd_sig = 1 if macd_h > 0 else (-1 if macd_h < 0 else 0)
        sr_range = sr["resistance"] - sr["support"]
        sr_sig   = 0
        if sr_range > 0:
            if curr < sr["support"]    + sr_range * 0.1: sr_sig =  1
            if curr > sr["resistance"] - sr_range * 0.1: sr_sig = -1
        ma_sig   = 1 if (curr > ma20 > ma50 and curr > ma200) else (-1 if (curr < ma20 < ma50 and curr < ma200) else 0)

        signals = {"RSI": rsi_sig, "MACD": macd_sig, "S/R": sr_sig, "Pattern": pattern, "MA": ma_sig}
        buy_s   = sum(1 for v in signals.values() if v ==  1)
        sell_s  = sum(1 for v in signals.values() if v == -1)

        direction = "BUY" if buy_s >= MIN_SCORE else "SELL" if sell_s >= MIN_SCORE else "WAIT"
        score     = buy_s if direction == "BUY" else sell_s if direction == "SELL" else max(buy_s, sell_s)
        price_str = f"{curr:.2f}" if curr > 100 else f"{curr:.4f}" if curr > 1 else f"{curr:.6f}"

        return {
            "symbol":     symbol,
            "price":      price_str,
            "direction":  direction,
            "score":      min(5, score),
            "rsi":        rsi,
            "support":    round(sr["support"],    6),
            "resistance": round(sr["resistance"], 6),
            "signals":    signals,
        }
    except Exception as e:
        logger.error(f"Analyze error {symbol}: {e}")
        return None

# ─── TELEGRAM ────────────────────────────────────────────────────

async def send_signal(app, r):
    if r["direction"] == "WAIT":
        return
    emoji = "🟢" if r["direction"] == "BUY" else "🔴"
    sym   = r["symbol"].replace("-", "/")
    def f(v): return "✅" if v == 1 else ("❌" if v == -1 else "⚪")
    s = r["signals"]
    msg = (
        f"{emoji} *{r['direction']} SIGNAL* — `{sym}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price: `${r['price']}`\n"
        f"⭐ Score: `{r['score']}/5`\n"
        f"📊 RSI:   `{r['rsi']}`\n\n"
        f"*Indicators:*\n"
        f"  RSI:     {f(s['RSI'])}\n"
        f"  MACD:    {f(s['MACD'])}\n"
        f"  S/R:     {f(s['S/R'])}\n"
        f"  Pattern: {f(s['Pattern'])}\n"
        f"  MA:      {f(s['MA'])}\n\n"
        f"📈 Support:    `${r['support']}`\n"
        f"📉 Resistance: `${r['resistance']}`\n"
        f"⏰ `{datetime.now().strftime('%H:%M:%S')}`"
    )
    keyboard = [[
        InlineKeyboardButton("❌ Skip", callback_data="skip"),
    ]]
    await app.bot.send_message(
        chat_id=TELEGRAM_CHAT_ID, text=msg,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    logger.info(f"Signal: {r['direction']} {r['symbol']} {r['score']}/5")

async def start_cmd(update, context):
    await update.message.reply_text(
        "🤖 *CryptoSignal Bot — OKX*\n\n"
        "/scan — Scan now\n/price — Prices\n/status — Status\n\n"
        "✅ Auto scan every 15 min!",
        parse_mode="Markdown"
    )

async def scan_cmd(update, context):
    await update.message.reply_text("🔍 Scanning 20 coins...")
    found = 0
    for sym in SYMBOLS:
        res = analyze(sym)
        if res and res["direction"] != "WAIT":
            await send_signal(context.application, res)
            found += 1
            await asyncio.sleep(0.5)
    msg = f"✅ Done — {found} signal(s)!" if found else "⚪ No strong signals right now."
    await update.message.reply_text(msg)

async def price_cmd(update, context):
    await update.message.reply_text("📊 Fetching prices...")
    msg = "💰 *Prices (OKX):*\n\n"
    for sym in SYMBOLS[:10]:
        t = get_ticker(sym)
        if t:
            name  = sym.replace("-USDT", "")
            p     = t["price"]
            ps    = f"{p:.2f}" if p > 100 else f"{p:.4f}" if p > 1 else f"{p:.6f}"
            chg   = t["change"]
            emoji = "🟢" if chg >= 0 else "🔴"
            msg  += f"{emoji} `{name}`: ${ps} ({chg:+.2f}%)\n"
        await asyncio.sleep(0.1)
    await update.message.reply_text(msg, parse_mode="Markdown")

async def status_cmd(update, context):
    await update.message.reply_text(
        f"✅ *RUNNING*\n📊 OKX | 🪙 {len(SYMBOLS)} coins | ⏱ M15\n"
        f"🎯 Min score: {MIN_SCORE}/5 | 🔄 Every 15 min",
        parse_mode="Markdown"
    )

async def button_cb(update, context):
    await update.callback_query.answer()
    await update.callback_query.edit_message_reply_markup(reply_markup=None)

async def auto_scan(app):
    await asyncio.sleep(20)
    while True:
        logger.info("Auto scan...")
        for sym in SYMBOLS:
            try:
                res = analyze(sym)
                if res and res["direction"] != "WAIT":
                    await send_signal(app, res)
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error {sym}: {e}")
        await asyncio.sleep(900)

async def post_init(app):
    asyncio.create_task(auto_scan(app))

def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID!")
        return
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start",  start_cmd))
    app.add_handler(CommandHandler("scan",   scan_cmd))
    app.add_handler(CommandHandler("price",  price_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("help",   start_cmd))
    app.add_handler(CallbackQueryHandler(button_cb))
    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()