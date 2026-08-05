"""
Сигналы РФ — одноразовый запуск для GitHub Actions.
Расписание: 10:30 МСК, пн–пт + проверка торгового дня ММВБ.
Секреты: BOT_TOKEN, CHAT_ID (env / GitHub Secrets).
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from telegram import Bot
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

MSK = ZoneInfo("Europe/Moscow")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Только из GitHub Secrets / переменных окружения — в коде ключей нет
BOT_TOKEN = (os.environ.get("BOT_TOKEN") or "").strip()
CHAT_ID = (os.environ.get("CHAT_ID") or "").strip()

SECURITIES = [
    "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "MTSS", "MGNT", "OZON", "TATN",
    "SNGS", "PIKK", "ALRS", "RUAL", "CHMF", "NLMK", "VTBR", "AFKS", "AFLT", "PHOR",
    "HYDR", "ENPG", "MOEX", "SNGSP", "MAGN", "TRMK", "BANEP", "RTKM", "SIBN", "RNFT",
]

DATA_FILE = DATA_DIR / "moex_data.xlsx"
SIGNALS_FILE = DATA_DIR / "signals.xlsx"
TRADES_FILE = DATA_DIR / "trades.xlsx"
REPORT_FILE = DATA_DIR / "weekly_report.xlsx"
RF_PATH = ROOT / "rf_model.pkl"
NN_PATH = ROOT / "nn_model.h5"
SCALER_PATH = ROOT / "scaler.pkl"

DAYS_TO_LOAD = 30
API_PAUSE = float(os.environ.get("API_PAUSE", "1.0"))
REQUEST_TIMEOUT = 30


def now_msk() -> datetime:
    return datetime.now(MSK)


def is_moex_trading_day(day: datetime | None = None) -> bool:
    """Пн–пт + TRADINGSTATUS TQBR (SBER). Праздники ММВБ отсекаются."""
    day = day or now_msk()
    if day.weekday() >= 5:
        print(f"Выходной ({day.date()}), пропуск.")
        return False

    try:
        url = (
            "https://iss.moex.com/iss/engines/stock/markets/shares/"
            "boards/TQBR/securities/SBER.json"
        )
        r = requests.get(
            url,
            params={"iss.meta": "off", "iss.only": "marketdata"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        md = r.json().get("marketdata", {})
        cols = md.get("columns") or []
        rows = md.get("data") or []
        if not rows:
            print(f"Нет marketdata SBER — неторговый день ({day.date()}).")
            return False
        row = dict(zip(cols, rows[0]))
        status = str(row.get("TRADINGSTATUS") or "").upper()
        num_trades = row.get("NUMTRADES") or 0
        last = row.get("LAST")
        # T = торги идут; до открытия иногда бывает пусто — тогда смотрим будни
        if status == "T" or (last is not None and num_trades and int(num_trades) > 0):
            print(
                f"Торговый день ММВБ: TRADINGSTATUS={status}, "
                f"LAST={last}, NUMTRADES={num_trades}"
            )
            return True
        if status in {"N", "C", "Z"}:
            print(f"Биржа закрыта (TRADINGSTATUS={status}) — пропуск.")
            return False
        # Утренний запуск около 10:30: если статус ещё не T, но будни —
        # всё равно работаем (открытие в 10:00)
        print(f"Статус {status!r}, будний день — продолжаем.")
        return True
    except Exception as e:
        print(f"Проверка торгового дня не удалась ({e}). Будни — продолжаем.")
        return True


def get_moex_historical_data(secid: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"История {secid}...")
    url = (
        f"https://iss.moex.com/iss/history/engines/stock/markets/shares/"
        f"boards/TQBR/securities/{secid}.json"
    )
    params = {
        "from": start_date,
        "till": end_date,
        "interval": 24,
        "iss.meta": "off",
        "iss.json": "extended",
        "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE,WAPRICE",
    }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) < 2:
            return pd.DataFrame()
        history_data = data[1].get("history", [])
        if not history_data:
            return pd.DataFrame()
        columns = params["history.columns"].split(",")
        df = pd.DataFrame(history_data, columns=columns)
        df["SECID"] = secid
        return df
    except Exception as e:
        print(f"Ошибка истории {secid}: {e}")
        return pd.DataFrame()


def get_moex_intraday_data(secid: str, date: str) -> pd.DataFrame:
    print(f"Внутридневные {secid}...")
    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/"
        f"boards/TQBR/securities/{secid}.json"
    )
    params = {
        "iss.meta": "off",
        "iss.json": "extended",
        "marketdata.columns": "OPEN,HIGH,LOW,LAST,VOLTODAY,VALTODAY_RUR,WAPRICE",
    }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) < 2:
            return pd.DataFrame()
        market_data = data[1].get("marketdata", [])
        if not market_data or not market_data[0]:
            return pd.DataFrame()
        df = pd.DataFrame(market_data, columns=params["marketdata.columns"].split(","))
        df["TRADEDATE"] = pd.to_datetime(date)
        df["SECID"] = secid
        df = df.rename(
            columns={"LAST": "CLOSE", "VOLTODAY": "VOLUME", "VALTODAY_RUR": "VALUE"}
        )
        return df[
            ["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE", "WAPRICE", "SECID"]
        ]
    except Exception as e:
        print(f"Ошибка внутридневных {secid}: {e}")
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("Индикаторы...")
    df = df.copy()
    df["SMA10"] = df.groupby("SECID")["CLOSE"].rolling(10).mean().reset_index(level=0, drop=True)
    df["SMA20"] = df.groupby("SECID")["CLOSE"].rolling(20).mean().reset_index(level=0, drop=True)
    delta = df.groupby("SECID")["CLOSE"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))
    df["LASTCHANGEPRCNT"] = df.groupby("SECID")["OPEN"].pct_change() * 100
    df["LASTCHANGEPRCNT"] = df["LASTCHANGEPRCNT"].fillna(0)
    df["AVG_VOLUME"] = (
        df.groupby("SECID")["VOLUME"].rolling(5).mean().reset_index(level=0, drop=True)
    )
    return df


def train_model(data: pd.DataFrame):
    print("Обучение моделей...")
    X = data[["LASTCHANGEPRCNT", "VOLUME", "WAPRICE", "SMA10", "SMA20", "RSI14"]].fillna(0)
    if "Signal" not in data.columns:
        data = data.copy()
        data["Signal"] = "Hold"
    y = data["Signal"].map({"Buy": 1, "Sell": 2, "Hold": 0}).fillna(0)

    unique_classes = np.unique(y)
    for class_label in [0, 1, 2]:
        if class_label not in unique_classes:
            for secid in SECURITIES:
                subset = data[data["SECID"] == secid]
                if subset.empty:
                    continue
                dummy_rows = subset.sample(min(5, len(subset)), replace=True)
                for _, dummy_row in dummy_rows.iterrows():
                    dummy_x = dummy_row[
                        ["LASTCHANGEPRCNT", "VOLUME", "WAPRICE", "SMA10", "SMA20", "RSI14"]
                    ]
                    X = pd.concat([X, pd.DataFrame([dummy_x])], ignore_index=True)
                    y = pd.concat([y, pd.Series([class_label])], ignore_index=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf_model.fit(X_scaled, y)

    nn_model = Sequential(
        [
            Input(shape=(X.shape[1],)),
            Dense(100, activation="relu"),
            Dense(50, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )
    nn_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    nn_model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)

    joblib.dump(rf_model, RF_PATH)
    nn_model.save(NN_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Модели сохранены")
    return rf_model, nn_model, scaler


def generate_signals(df, rf_model, nn_model, scaler) -> pd.DataFrame:
    print("Генерация сигналов...")
    signals = []
    today = df["TRADEDATE"].max()
    today_data = df[df["TRADEDATE"] == today].copy().reset_index(drop=True)
    if today_data.empty:
        return pd.DataFrame()

    X = today_data[["LASTCHANGEPRCNT", "VOLUME", "WAPRICE", "SMA10", "SMA20", "RSI14"]].fillna(0)
    X_scaled = scaler.transform(X)
    rf_probs = rf_model.predict_proba(X_scaled)
    nn_probs = nn_model.predict(X_scaled, verbose=0)
    ensemble_probs = (rf_probs + nn_probs) / 2

    for idx, row in today_data.iterrows():
        signal = "Hold"
        prob = ensemble_probs[idx]
        buy_prob = float(prob[1])
        sell_prob = float(prob[2])

        if (
            row["LASTCHANGEPRCNT"] > 0.1
            and row["VOLUME"] > row["AVG_VOLUME"]
            and row["OPEN"] > row["SMA10"]
            and row["SMA10"] > row["SMA20"]
            and 15 <= row["RSI14"] <= 85
        ):
            signal = "Buy"
        elif (
            row["OPEN"] < row["WAPRICE"] * 0.98
            and row["LASTCHANGEPRCNT"] > 0
            and row["VOLUME"] > row["AVG_VOLUME"]
            and row["RSI14"] < 30
        ):
            signal = "Buy"
        elif (
            row["LASTCHANGEPRCNT"] < -0.1
            or row["OPEN"] < row["SMA10"]
            or row["OPEN"] > row["WAPRICE"] * 1.02
            or row["RSI14"] > 85
        ):
            signal = "Sell"

        if signal == "Buy":
            stop_loss = row["OPEN"] * 0.95
            take_profit = row["OPEN"] * 1.08
        elif signal == "Sell":
            stop_loss = row["OPEN"] * 1.05
            take_profit = row["OPEN"] * 0.95
        else:
            stop_loss = row["OPEN"] * 0.95
            take_profit = row["OPEN"] * 1.08

        signals.append(
            {
                "TRADEDATE": row["TRADEDATE"],
                "SECID": row["SECID"],
                "Signal": signal,
                "Price": row["OPEN"],
                "Stop_Loss": stop_loss,
                "Take_Profit": take_profit,
                "Buy_Prob": buy_prob,
                "Sell_Prob": sell_prob,
                "LASTCHANGEPRCNT": row["LASTCHANGEPRCNT"],
            }
        )

    return pd.DataFrame(signals)


async def send_telegram_message(bot: Bot, chat_id, text: str) -> None:
    print(f"TG: {text[:80]}...")
    try:
        await bot.send_message(chat_id=chat_id, text=text)
        print("TG OK")
    except Exception as e:
        print(f"TG ошибка: {e}")
        raise


async def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        raise SystemExit(
            "Задайте BOT_TOKEN и CHAT_ID (переменные окружения / GitHub Secrets)."
        )

    today = now_msk()
    print(f"Запуск {today.isoformat()}")

    if not is_moex_trading_day(today):
        print(f"{today.date()}: неторговый день ММВБ — выход без работы.")
        return

    bot = Bot(token=BOT_TOKEN)
    print("Торговый день — собираю данные...")

    all_data = []
    if DATA_FILE.exists():
        existing_data = pd.read_excel(DATA_FILE)
        all_data.append(existing_data)
        for i, secid in enumerate(SECURITIES):
            df_today = get_moex_intraday_data(secid, today.strftime("%Y-%m-%d"))
            if not df_today.empty:
                all_data = [
                    df
                    for df in all_data
                    if not (
                        (df["SECID"] == secid).any()
                        and (pd.to_datetime(df["TRADEDATE"]).dt.normalize() == pd.Timestamp(today.date())).any()
                    )
                ]
                all_data.append(df_today)
            time.sleep(API_PAUSE)
            print(f"Обработано {i + 1}/{len(SECURITIES)}")
    else:
        start_date = today - timedelta(days=DAYS_TO_LOAD)
        for i, secid in enumerate(SECURITIES):
            df_hist = get_moex_historical_data(
                secid,
                start_date.strftime("%Y-%m-%d"),
                (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if not df_hist.empty:
                all_data.append(df_hist)
            df_today = get_moex_intraday_data(secid, today.strftime("%Y-%m-%d"))
            if not df_today.empty:
                all_data.append(df_today)
            time.sleep(API_PAUSE)
            print(f"Обработано {i + 1}/{len(SECURITIES)}")

    if not all_data:
        await send_telegram_message(
            bot, CHAT_ID, "Ошибка: не удалось загрузить данные MOEX."
        )
        return

    df = pd.concat(all_data, ignore_index=True)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values(["TRADEDATE", "SECID"]).drop_duplicates(
        subset=["TRADEDATE", "SECID"], keep="last"
    )
    df.to_excel(DATA_FILE, index=False)

    df = calculate_indicators(df)

    if SIGNALS_FILE.exists():
        existing_signals = pd.read_excel(SIGNALS_FILE)
        if "TRADEDATE" not in existing_signals.columns:
            existing_signals["TRADEDATE"] = pd.to_datetime("1970-01-01")
        existing_signals["TRADEDATE"] = pd.to_datetime(existing_signals["TRADEDATE"])
        if "Signal" not in existing_signals.columns:
            existing_signals["Signal"] = "Hold"
        signals_index = existing_signals.set_index(["TRADEDATE", "SECID"])["Signal"]
        df_index = df.set_index(["TRADEDATE", "SECID"])
        df["Signal"] = signals_index.reindex(df_index.index, fill_value="Hold").values
    else:
        existing_signals = pd.DataFrame(
            columns=[
                "TRADEDATE",
                "SECID",
                "Signal",
                "Price",
                "Stop_Loss",
                "Take_Profit",
                "Buy_Prob",
                "LASTCHANGEPRCNT",
            ]
        )
        df["Signal"] = "Hold"

    if RF_PATH.exists() and NN_PATH.exists() and SCALER_PATH.exists():
        print("Загрузка сохранённых моделей...")
        rf_model = joblib.load(RF_PATH)
        nn_model = tf.keras.models.load_model(NN_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        rf_model, nn_model, scaler = train_model(df)

    signals = generate_signals(df, rf_model, nn_model, scaler)

    if not signals.empty:
        if existing_signals.empty:
            new_signals = signals
        else:
            new_signals = pd.concat([existing_signals, signals], ignore_index=True)
        new_signals = new_signals.sort_values(["TRADEDATE", "SECID"]).drop_duplicates(
            subset=["TRADEDATE", "SECID"], keep="last"
        )
        new_signals.to_excel(SIGNALS_FILE, index=False)

    trades = []
    if TRADES_FILE.exists():
        trades_df = pd.read_excel(TRADES_FILE)
        trades_df["Entry_Date"] = pd.to_datetime(trades_df["Entry_Date"])
        trades_df["Exit_Date"] = pd.to_datetime(trades_df["Exit_Date"], errors="coerce")
        trades = trades_df.to_dict("records")

    open_positions = [t for t in trades if pd.isna(t.get("Exit_Date"))]
    today_naive = today.replace(tzinfo=None)

    for pos in open_positions[:]:
        secid = pos["SECID"]
        entry_price = pos["Entry_Price"]
        today_data = df[(df["SECID"] == secid) & (df["TRADEDATE"] == df["TRADEDATE"].max())]
        if today_data.empty:
            continue
        high, low, close = (
            today_data["HIGH"].iloc[0],
            today_data["LOW"].iloc[0],
            today_data["CLOSE"].iloc[0],
        )
        stop_loss = pos["Stop_Loss"]
        take_profit = pos["Take_Profit"]
        entry_date = pd.to_datetime(pos["Entry_Date"])
        signal = pos.get("Signal", "Buy")
        position_closed = False

        if signal == "Buy":
            if high >= take_profit:
                exit_price = take_profit
                profit = (exit_price - entry_price) / entry_price * 100
                pos["Exit_Date"] = today_naive
                pos["Exit_Price"] = exit_price
                pos["Profit_Loss"] = profit
                await send_telegram_message(
                    bot,
                    CHAT_ID,
                    f"{today.date()}, {secid}: Закрыт по тейк-профиту\n"
                    f"Цена входа: {entry_price}\nЦена выхода: {exit_price}\n"
                    f"Прибыль: {profit:.2f}%",
                )
                position_closed = True
            elif low <= stop_loss:
                exit_price = stop_loss
                profit = (exit_price - entry_price) / entry_price * 100
                pos["Exit_Date"] = today_naive
                pos["Exit_Price"] = exit_price
                pos["Profit_Loss"] = profit
                await send_telegram_message(
                    bot,
                    CHAT_ID,
                    f"{today.date()}, {secid}: Закрыт по стоп-лоссу\n"
                    f"Цена входа: {entry_price}\nЦена выхода: {exit_price}\n"
                    f"Убыток: {profit:.2f}%",
                )
                position_closed = True
        elif signal == "Sell":
            if low <= take_profit:
                exit_price = take_profit
                profit = (entry_price - exit_price) / entry_price * 100
                pos["Exit_Date"] = today_naive
                pos["Exit_Price"] = exit_price
                pos["Profit_Loss"] = profit
                await send_telegram_message(
                    bot,
                    CHAT_ID,
                    f"{today.date()}, {secid}: Закрыт по тейк-профиту (шорт)\n"
                    f"Цена входа: {entry_price}\nЦена выхода: {exit_price}\n"
                    f"Прибыль: {profit:.2f}%",
                )
                position_closed = True
            elif high >= stop_loss:
                exit_price = stop_loss
                profit = (entry_price - exit_price) / entry_price * 100
                pos["Exit_Date"] = today_naive
                pos["Exit_Price"] = exit_price
                pos["Profit_Loss"] = profit
                await send_telegram_message(
                    bot,
                    CHAT_ID,
                    f"{today.date()}, {secid}: Закрыт по стоп-лоссу (шорт)\n"
                    f"Цена входа: {entry_price}\nЦена выхода: {exit_price}\n"
                    f"Убыток: {profit:.2f}%",
                )
                position_closed = True

        if (today_naive - entry_date.to_pydatetime().replace(tzinfo=None)).days >= 3:
            exit_price = close
            if signal == "Buy":
                profit = (exit_price - entry_price) / entry_price * 100
            else:
                profit = (entry_price - exit_price) / entry_price * 100
            pos["Exit_Date"] = today_naive
            pos["Exit_Price"] = exit_price
            pos["Profit_Loss"] = profit
            await send_telegram_message(
                bot,
                CHAT_ID,
                f"{today.date()}, {secid}: Закрыт по времени (3 дня)\n"
                f"Цена входа: {entry_price}\nЦена выхода: {exit_price}\n"
                f"Прибыль/Убыток: {profit:.2f}%",
            )
            position_closed = True

        if position_closed:
            open_positions.remove(pos)

    buy_signals = (
        signals[signals["Signal"] == "Buy"].sort_values("Buy_Prob", ascending=False).head(5)
        if not signals.empty
        else signals
    )
    sell_signals = (
        signals[signals["Signal"] == "Sell"].sort_values("Buy_Prob", ascending=True).head(5)
        if not signals.empty
        else signals
    )

    if not buy_signals.empty:
        for _, row in buy_signals.iterrows():
            text = (
                f"{row['TRADEDATE'].date()}, {row['SECID']}: Покупка\n"
                f"Цена: {row['Price']:.2f}\n"
                f"Стоп-лосс: {row['Stop_Loss']:.2f}\n"
                f"Тейк-профит: {row['Take_Profit']:.2f}\n"
                f"Причина: {'Импульс' if row['LASTCHANGEPRCNT'] > 0.1 else 'Реверсия'}"
            )
            await send_telegram_message(bot, CHAT_ID, text)
            trades.append(
                {
                    "Entry_Date": row["TRADEDATE"],
                    "SECID": row["SECID"],
                    "Entry_Price": row["Price"],
                    "Stop_Loss": row["Stop_Loss"],
                    "Take_Profit": row["Take_Profit"],
                    "Signal": row["Signal"],
                    "Exit_Date": None,
                    "Exit_Price": None,
                    "Profit_Loss": None,
                }
            )

    if not sell_signals.empty:
        for _, row in sell_signals.iterrows():
            text = (
                f"{row['TRADEDATE'].date()}, {row['SECID']}: Продажа (шорт)\n"
                f"Цена: {row['Price']:.2f}\n"
                f"Стоп-лосс: {row['Stop_Loss']:.2f}\n"
                f"Тейк-профит: {row['Take_Profit']:.2f}\n"
                f"Причина: {'Падение' if row['LASTCHANGEPRCNT'] < -0.1 else 'Перекупленность'}"
            )
            await send_telegram_message(bot, CHAT_ID, text)
            trades.append(
                {
                    "Entry_Date": row["TRADEDATE"],
                    "SECID": row["SECID"],
                    "Entry_Price": row["Price"],
                    "Stop_Loss": row["Stop_Loss"],
                    "Take_Profit": row["Take_Profit"],
                    "Signal": row["Signal"],
                    "Exit_Date": None,
                    "Exit_Price": None,
                    "Profit_Loss": None,
                }
            )

    if (buy_signals.empty if hasattr(buy_signals, "empty") else True) and (
        sell_signals.empty if hasattr(sell_signals, "empty") else True
    ):
        await send_telegram_message(
            bot, CHAT_ID, f"{today.date()}: Нет сигналов для торговли"
        )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.drop_duplicates(
            subset=["Entry_Date", "SECID", "Entry_Price"], keep="last"
        )
    trades_df.to_excel(TRADES_FILE, index=False)

    if today.weekday() == 4:
        report = pd.DataFrame([t for t in trades if not pd.isna(t.get("Exit_Date"))])
        if not report.empty:
            report.to_excel(REPORT_FILE, index=False)
            with open(REPORT_FILE, "rb") as document:
                await bot.send_document(
                    chat_id=CHAT_ID,
                    document=document,
                    caption=f"Недельный отчет за {today.date()}",
                )

    await send_telegram_message(
        bot, CHAT_ID, f"{today.date()}: цикл Сигналы РФ завершён."
    )
    print("Готово.")


if __name__ == "__main__":
    asyncio.run(main())
