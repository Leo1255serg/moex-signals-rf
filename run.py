import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from telegram import Bot
import asyncio
import os
import joblib
import tensorflow as tf

# Настройки
BOT_TOKEN = "7831097053:AAE5lFirdDiDbdCa45eLh3k5tuYrWYLVS00"
CHAT_ID = 283220567
SECURITIES = [
    'SBER', 'GAZP', 'LKOH', 'ROSN', 'NVTK', 'GMKN', 'MTSS', 'MGNT', 'OZON', 'TATN',
    'SNGS', 'PIKK', 'ALRS', 'RUAL', 'CHMF', 'NLMK', 'VTBR', 'AFKS', 'AFLT', 'PHOR',
    'HYDR', 'ENPG', 'MOEX', 'SNGSP', 'MAGN', 'TRMK', 'BANEP', 'RTKM', 'SIBN', 'RNFT'
]
DATA_FILE = "moex_data.xlsx"
SIGNALS_FILE = "signals.xlsx"
TRADES_FILE = "trades.xlsx"
REPORT_FILE = "weekly_report.xlsx"
DAYS_TO_LOAD = 30

# Функция для получения исторических данных
def get_moex_historical_data(secid, start_date, end_date):
    print(f"Загружаю исторические данные для {secid}...")
    url = f"http://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{secid}.json"
    params = {
        'from': start_date,
        'till': end_date,
        'interval': 24,
        'iss.meta': 'off',
        'iss.json': 'extended',
        'history.columns': 'TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE,WAPRICE'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) < 2:
            print(f"Ошибка: Неправильный формат данных для {secid}")
            return pd.DataFrame()
        history_data = data[1].get('history', [])
        if not history_data:
            print(f"Нет исторических данных для {secid}")
            return pd.DataFrame()
        columns = params['history.columns'].split(',')
        df = pd.DataFrame(history_data, columns=columns)
        df['SECID'] = secid
        print(f"Успешно загружено {len(df)} строк для {secid}")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных для {secid}: {e}")
        return pd.DataFrame()

# Функция для получения внутридневных данных
def get_moex_intraday_data(secid, date):
    print(f"Загружаю внутридневные данные для {secid} на {date}...")
    url = f"http://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{secid}.json"
    params = {
        'iss.meta': 'off',
        'iss.json': 'extended',
        'marketdata.columns': 'OPEN,HIGH,LOW,LAST,VOLTODAY,VALTODAY_RUR,WAPRICE'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) < 2:
            print(f"Ошибка: Неправильный формат внутридневных данных для {secid}")
            return pd.DataFrame()
        market_data = data[1].get('marketdata', [])
        if not market_data or not market_data[0]:
            print(f"Нет внутридневных данных для {secid}")
            return pd.DataFrame()
        df = pd.DataFrame(market_data, columns=params['marketdata.columns'].split(','))
        df['TRADEDATE'] = pd.to_datetime(date)
        df['SECID'] = secid
        df = df.rename(columns={'LAST': 'CLOSE', 'VOLTODAY': 'VOLUME', 'VALTODAY_RUR': 'VALUE'})
        print(f"Успешно загружено внутридневных данных для {secid}")
        return df[['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'VALUE', 'WAPRICE', 'SECID']]
    except Exception as e:
        print(f"Ошибка при загрузке внутридневных данных для {secid}: {e}")
        return pd.DataFrame()

# Функция для расчета индикаторов
def calculate_indicators(df):
    print("Рассчитываю индикаторы...")
    df['SMA10'] = df.groupby('SECID')['CLOSE'].rolling(10).mean().reset_index(level=0, drop=True)
    df['SMA20'] = df.groupby('SECID')['CLOSE'].rolling(20).mean().reset_index(level=0, drop=True)
    delta = df.groupby('SECID')['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    df['LASTCHANGEPRCNT'] = df.groupby('SECID')['OPEN'].pct_change() * 100
    df['LASTCHANGEPRCNT'] = df['LASTCHANGEPRCNT'].fillna(0)
    df['AVG_VOLUME'] = df.groupby('SECID')['VOLUME'].rolling(5).mean().reset_index(level=0, drop=True)
    print("Индикаторы рассчитаны")
    return df

# Функция для обучения ML/NN модели
def train_model(data):
    print("Обучаю ML/NN модели...")
    X = data[['LASTCHANGEPRCNT', 'VOLUME', 'WAPRICE', 'SMA10', 'SMA20', 'RSI14']].fillna(0)
    
    # Проверяем, есть ли столбец 'Signal', если нет — создаём с дефолтным значением 'Hold'
    if 'Signal' not in data.columns:
        print("Столбец 'Signal' не найден, создаю с значением 'Hold'...")
        data['Signal'] = 'Hold'
    
    y = data['Signal'].map({'Buy': 1, 'Sell': 2, 'Hold': 0}).fillna(0)
    
    # Проверяем наличие всех классов в y
    unique_classes = np.unique(y)
    print(f"Классы в y перед обучением: {unique_classes}")
    
    # Добавляем фиктивные данные для каждого класса для всех акций
    for class_label in [0, 1, 2]:  # Hold, Buy, Sell
        if class_label not in unique_classes:
            print(f"Добавляю фиктивные данные для класса {class_label}")
            for secid in SECURITIES:
                dummy_rows = data[data['SECID'] == secid].sample(min(5, len(data[data['SECID'] == secid])), replace=True)
                for _, dummy_row in dummy_rows.iterrows():
                    dummy_x = dummy_row[['LASTCHANGEPRCNT', 'VOLUME', 'WAPRICE', 'SMA10', 'SMA20', 'RSI14']]
                    X = pd.concat([X, pd.DataFrame([dummy_x])], ignore_index=True)
                    y = pd.concat([y, pd.Series([class_label])], ignore_index=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_scaled, y)
    
    # NN
    nn_model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(3, activation='softmax')
    ])
    nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)
    
    # Сохранение моделей
    print("Сохраняю модели...")
    joblib.dump(rf_model, 'rf_model.pkl')  # Сохраняем Random Forest
    nn_model.save('nn_model.h5')  # Сохраняем нейросеть
    joblib.dump(scaler, 'scaler.pkl')  # Сохраняем scaler
    
    print("Модели обучены и сохранены")
    return rf_model, nn_model, scaler

# Функция для генерации сигналов
def generate_signals(df, rf_model, nn_model, scaler):
    print("Генерирую сигналы...")
    signals = []
    today = df['TRADEDATE'].max()
    today_data = df[df['TRADEDATE'] == today].copy().reset_index(drop=True)
    
    if today_data.empty:
        print("Нет данных за текущий день")
        return pd.DataFrame()
    
    X = today_data[['LASTCHANGEPRCNT', 'VOLUME', 'WAPRICE', 'SMA10', 'SMA20', 'RSI14']].fillna(0)
    X_scaled = scaler.transform(X)
    
    rf_probs = rf_model.predict_proba(X_scaled)
    nn_probs = nn_model.predict(X_scaled, verbose=0)
    print(f"Размер rf_probs: {rf_probs.shape}, nn_probs: {nn_probs.shape}")
    ensemble_probs = (rf_probs + nn_probs) / 2
    
    for idx, row in today_data.iterrows():
        signal = 'Hold'
        prob = ensemble_probs[idx]
        buy_prob = prob[1]
        sell_prob = prob[2]
        
        # Импульс
        if (row['LASTCHANGEPRCNT'] > 0.1 and
            row['VOLUME'] > row['AVG_VOLUME'] and
            row['OPEN'] > row['SMA10'] and
            row['SMA10'] > row['SMA20'] and
            15 <= row['RSI14'] <= 85):
            signal = 'Buy'
        # Реверсия
        elif (row['OPEN'] < row['WAPRICE'] * 0.98 and
              row['LASTCHANGEPRCNT'] > 0 and
              row['VOLUME'] > row['AVG_VOLUME'] and
              row['RSI14'] < 30):
            signal = 'Buy'
        # Продажа
        elif (row['LASTCHANGEPRCNT'] < -0.1 or
              row['OPEN'] < row['SMA10'] or
              row['OPEN'] > row['WAPRICE'] * 1.02 or
              row['RSI14'] > 85):
            signal = 'Sell'
        
        # Расчет Stop_Loss и Take_Profit в зависимости от сигнала
        if signal == 'Buy':
            stop_loss = row['OPEN'] * 0.95
            take_profit = row['OPEN'] * 1.08
        elif signal == 'Sell':
            stop_loss = row['OPEN'] * 1.05
            take_profit = row['OPEN'] * 0.95
        else:  # Hold
            stop_loss = row['OPEN'] * 0.95
            take_profit = row['OPEN'] * 1.08
        
        signals.append({
            'TRADEDATE': row['TRADEDATE'],
            'SECID': row['SECID'],
            'Signal': signal,
            'Price': row['OPEN'],
            'Stop_Loss': stop_loss,
            'Take_Profit': take_profit,
            'Buy_Prob': buy_prob,
            'LASTCHANGEPRCNT': row['LASTCHANGEPRCNT']
        })
    
    print(f"Сгенерировано {len(signals)} сигналов")
    return pd.DataFrame(signals)

# Функция для отправки сообщений в Telegram (исправлено)
async def send_telegram_message(bot, chat_id, text):
    print(f"Отправляю сообщение в Telegram: {text[:50]}...")
    try:
        await bot.send_message(chat_id=chat_id, text=text)
        print("Сообщение успешно отправлено")
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")

# Основная функция
async def main():
    bot = Bot(token=BOT_TOKEN)
    print("Запуск системы...")
    
    # Сбор данных
    today = datetime.now()
    
    all_data = []
    if os.path.exists(DATA_FILE):
        print(f"Загружаю существующие данные из {DATA_FILE}...")
        existing_data = pd.read_excel(DATA_FILE)
        all_data.append(existing_data)
        for i, secid in enumerate(SECURITIES):
            df_today = get_moex_intraday_data(secid, today.strftime('%Y-%m-%d'))
            if not df_today.empty:
                # Удаляем старые данные за текущий день для этой акции
                all_data = [df for df in all_data if not (df['SECID'] == secid).any() or not (df['TRADEDATE'] == today).any()]
                all_data.append(df_today)
            time.sleep(5)
            print(f"Обработано {i+1}/{len(SECURITIES)} акций")
    else:
        start_date = today - timedelta(days=DAYS_TO_LOAD)
        for i, secid in enumerate(SECURITIES):
            df_hist = get_moex_historical_data(secid, start_date.strftime('%Y-%m-%d'), (today - timedelta(days=1)).strftime('%Y-%m-%d'))
            if not df_hist.empty:
                all_data.append(df_hist)
            df_today = get_moex_intraday_data(secid, today.strftime('%Y-%m-%d'))
            if not df_today.empty:
                all_data.append(df_today)
            time.sleep(5)
            print(f"Обработано {i+1}/{len(SECURITIES)} акций")
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df = df.sort_values(['TRADEDATE', 'SECID']).drop_duplicates(subset=['TRADEDATE', 'SECID'], keep='last')
        print(f"Сохраняю данные в {DATA_FILE}...")
        df.to_excel(DATA_FILE, index=False)
    else:
        print("Не удалось получить данные. Проверьте интернет или API.")
        await send_telegram_message(bot, CHAT_ID, "Ошибка: Не удалось загрузить данные. Проверьте интернет.")
        return
    
    # Расчет индикаторов
    df = calculate_indicators(df)
    
    # Загрузка существующих сигналов
    if os.path.exists(SIGNALS_FILE):
        print(f"Загружаю существующие сигналы из {SIGNALS_FILE}...")
        existing_signals = pd.read_excel(SIGNALS_FILE)
        
        # Проверяем наличие столбца 'TRADEDATE'
        if 'TRADEDATE' not in existing_signals.columns:
            print("Столбец 'TRADEDATE' отсутствует в signals.xlsx, создаю новый...")
            existing_signals['TRADEDATE'] = pd.to_datetime('1970-01-01')  # Устанавливаем дефолтную дату
        
        existing_signals['TRADEDATE'] = pd.to_datetime(existing_signals['TRADEDATE'])
        
        # Проверяем, что в existing_signals есть столбец 'Signal'
        if 'Signal' not in existing_signals.columns:
            print("Столбец 'Signal' отсутствует в signals.xlsx, добавляю с значением 'Hold'...")
            existing_signals['Signal'] = 'Hold'
        
        # Создаём индексы для выравнивания
        signals_index = existing_signals.set_index(['TRADEDATE', 'SECID'])['Signal']
        df_index = df.set_index(['TRADEDATE', 'SECID'])
        
        # Выравниваем сигналы с df
        df['Signal'] = signals_index.reindex(df_index.index, fill_value='Hold').values
    else:
        print("Файл signals.xlsx не найден, создаю новый...")
        existing_signals = pd.DataFrame(columns=['TRADEDATE', 'SECID', 'Signal', 'Price', 'Stop_Loss', 'Take_Profit', 'Buy_Prob', 'LASTCHANGEPRCNT'])
        df['Signal'] = 'Hold'  # Устанавливаем 'Hold' для всех строк
    
    # Обучение модели или загрузка сохранённых моделей
    if os.path.exists('rf_model.pkl') and os.path.exists('nn_model.h5') and os.path.exists('scaler.pkl'):
        print("Загружаю сохранённые модели...")
        rf_model = joblib.load('rf_model.pkl')
        nn_model = tf.keras.models.load_model('nn_model.h5')
        scaler = joblib.load('scaler.pkl')
    else:
        print("Модели не найдены, обучаю новые...")
        rf_model, nn_model, scaler = train_model(df)
    
    # Генерация сигналов
    signals = generate_signals(df, rf_model, nn_model, scaler)
    
    # Сохранение сигналов (добавляем, а не перезаписываем)
    if not signals.empty:
        if existing_signals.empty:
            new_signals = signals
        else:
            new_signals = pd.concat([existing_signals, signals], ignore_index=True)
        new_signals = new_signals.sort_values(['TRADEDATE', 'SECID']).drop_duplicates(subset=['TRADEDATE', 'SECID'], keep='last')
        print(f"Сохраняю сигналы в {SIGNALS_FILE}...")
        new_signals.to_excel(SIGNALS_FILE, index=False)
    
    # Загрузка существующих сделок
    trades = []
    if os.path.exists(TRADES_FILE):
        trades_df = pd.read_excel(TRADES_FILE)
        trades_df['Entry_Date'] = pd.to_datetime(trades_df['Entry_Date'])
        trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'], errors='coerce')
        trades = trades_df.to_dict('records')
    
    # Проверка закрытия позиций
    open_positions = [t for t in trades if pd.isna(t.get('Exit_Date'))]
    closed_positions = []  # Список для хранения закрытых позиций в этом запуске
    
    for pos in open_positions[:]:  # Делаем копию списка, чтобы можно было удалять элементы
        secid = pos['SECID']
        entry_price = pos['Entry_Price']
        today_data = df[(df['SECID'] == secid) & (df['TRADEDATE'] == df['TRADEDATE'].max())]
        if today_data.empty:
            continue
        high, low, close = today_data['HIGH'].iloc[0], today_data['LOW'].iloc[0], today_data['CLOSE'].iloc[0]
        stop_loss = pos['Stop_Loss']
        take_profit = pos['Take_Profit']
        entry_date = pd.to_datetime(pos['Entry_Date'])
        signal = pos.get('Signal', 'Buy')  # Предполагаем 'Buy', если сигнал не указан
        
        # Флаг, чтобы отметить, была ли позиция закрыта
        position_closed = False
        
        if signal == 'Buy':
            if high >= take_profit:
                exit_price = take_profit
                profit = (exit_price - entry_price) / entry_price * 100
                pos['Exit_Date'] = today
                pos['Exit_Price'] = exit_price
                pos['Profit_Loss'] = profit
                text = f"{today.date()}, {secid}: Закрыт по тейк-профиту\nЦена входа: {entry_price}\nЦена выхода: {exit_price}\nПрибыль: {profit:.2f}%"
                await send_telegram_message(bot, CHAT_ID, text)
                position_closed = True
            elif low <= stop_loss:
                exit_price = stop_loss
                profit = (exit_price - entry_price) / entry_price * 100
                pos['Exit_Date'] = today
                pos['Exit_Price'] = exit_price
                pos['Profit_Loss'] = profit
                text = f"{today.date()}, {secid}: Закрыт по стоп-лоссу\nЦена входа: {entry_price}\nЦена выхода: {exit_price}\nУбыток: {profit:.2f}%"
                await send_telegram_message(bot, CHAT_ID, text)
                position_closed = True
        elif signal == 'Sell':
            if low <= take_profit:
                exit_price = take_profit
                profit = (entry_price - exit_price) / entry_price * 100
                pos['Exit_Date'] = today
                pos['Exit_Price'] = exit_price
                pos['Profit_Loss'] = profit
                text = f"{today.date()}, {secid}: Закрыт по тейк-профиту (шорт)\nЦена входа: {entry_price}\nЦена выхода: {exit_price}\nПрибыль: {profit:.2f}%"
                await send_telegram_message(bot, CHAT_ID, text)
                position_closed = True
            elif high >= stop_loss:
                exit_price = stop_loss
                profit = (entry_price - exit_price) / entry_price * 100
                pos['Exit_Date'] = today
                pos['Exit_Price'] = exit_price
                pos['Profit_Loss'] = profit
                text = f"{today.date()}, {secid}: Закрыт по стоп-лоссу (шорт)\nЦена входа: {entry_price}\nЦена выхода: {exit_price}\nУбыток: {profit:.2f}%"
                await send_telegram_message(bot, CHAT_ID, text)
                position_closed = True
        
        # Проверка на закрытие по времени (3 дня)
        if (today - entry_date).days >= 3:
            exit_price = close
            if signal == 'Buy':
                profit = (exit_price - entry_price) / entry_price * 100
            else:
                profit = (entry_price - exit_price) / entry_price * 100
            pos['Exit_Date'] = today
            pos['Exit_Price'] = exit_price
            pos['Profit_Loss'] = profit
            text = f"{today.date()}, {secid}: Закрыт по времени (3 дня)\nЦена входа: {entry_price}\nЦена выхода: {exit_price}\nПрибыль/Убыток: {profit:.2f}%"
            await send_telegram_message(bot, CHAT_ID, text)
            position_closed = True
        
        # Если позиция закрыта, добавляем её в closed_positions и удаляем из open_positions
        if position_closed:
            closed_positions.append(pos)
            open_positions.remove(pos)
    
    # Отправка сигналов и создание новых трейдов
    buy_signals = signals[signals['Signal'] == 'Buy'].sort_values('Buy_Prob', ascending=False).head(5)
    sell_signals = signals[signals['Signal'] == 'Sell'].sort_values('Buy_Prob', ascending=True).head(5)
    
    # Покупка (длинные позиции)
    if not buy_signals.empty:
        for idx, row in buy_signals.iterrows():
            text = (f"{row['TRADEDATE'].date()}, {row['SECID']}: Покупка\n"
                    f"Цена: {row['Price']:.2f}\n"
                    f"Стоп-лосс: {row['Stop_Loss']:.2f}\n"
                    f"Тейк-профит: {row['Take_Profit']:.2f}\n"
                    f"Причина: {'Импульс' if row['LASTCHANGEPRCNT'] > 0.1 else 'Реверсия'}")
            await send_telegram_message(bot, CHAT_ID, text)
            trades.append({
                'Entry_Date': row['TRADEDATE'],
                'SECID': row['SECID'],
                'Entry_Price': row['Price'],
                'Stop_Loss': row['Stop_Loss'],
                'Take_Profit': row['Take_Profit'],
                'Signal': row['Signal'],
                'Exit_Date': None,
                'Exit_Price': None,
                'Profit_Loss': None
            })
    
    # Продажа (короткие позиции)
    if not sell_signals.empty:
        for idx, row in sell_signals.iterrows():
            text = (f"{row['TRADEDATE'].date()}, {row['SECID']}: Продажа (шорт)\n"
                    f"Цена: {row['Price']:.2f}\n"
                    f"Стоп-лосс: {row['Stop_Loss']:.2f}\n"
                    f"Тейк-профит: {row['Take_Profit']:.2f}\n"
                    f"Причина: {'Падение' if row['LASTCHANGEPRCNT'] < -0.1 else 'Перекупленность'}")
            await send_telegram_message(bot, CHAT_ID, text)
            trades.append({
                'Entry_Date': row['TRADEDATE'],
                'SECID': row['SECID'],
                'Entry_Price': row['Price'],
                'Stop_Loss': row['Stop_Loss'],
                'Take_Profit': row['Take_Profit'],
                'Signal': row['Signal'],
                'Exit_Date': None,
                'Exit_Price': None,
                'Profit_Loss': None
            })
    
    if buy_signals.empty and sell_signals.empty:
        await send_telegram_message(bot, CHAT_ID, f"{today.date()}: Нет сигналов для торговли")
    
    # Сохранение сделок
    print(f"Сохраняю сделки в {TRADES_FILE}...")
    trades_df = pd.DataFrame(trades)
    # Удаляем дубликаты сделок на основе Entry_Date, SECID и Entry_Price
    trades_df = trades_df.drop_duplicates(subset=['Entry_Date', 'SECID', 'Entry_Price'], keep='last')
    trades_df.to_excel(TRADES_FILE, index=False)
    
    # Недельный отчет (по пятницам)
    if today.weekday() == 4:
        report = pd.DataFrame([t for t in trades if not pd.isna(t.get('Exit_Date'))])
        if not report.empty:
            print(f"Сохраняю недельный отчет в {REPORT_FILE}...")
            report.to_excel(REPORT_FILE, index=False)
            with open(REPORT_FILE, 'rb') as document:
                await bot.send_document(chat_id=CHAT_ID, document=document, caption=f"Недельный отчет за {today.date()}")

# Запуск
if __name__ == "__main__":
    asyncio.run(main())
