import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm


#%% Download data from Binance
def get_klines(symbol, interval, start_date, end_date):
    url = "https://api.binance.com/api/v3/klines"  # Binance API
    # convert to millisecond timestamp
    start = int(pd.to_datetime(start_date).timestamp() * 1000)
    end = int(pd.to_datetime(end_date).timestamp() * 1000)
    # read data
    all_data = []
    while start < end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start,
            "endTime": end,
            "limit": 1000  # max number of klines is 1000 each read
        }
        res = requests.get(url, params=params)
        data = res.json()

        if not data:
            break

        all_data.extend(data)
        # renew start time: close time of last kline + 1
        start = data[-1][6] + 1  # [-1] to get the last kline in this read, [6] to get the 7th field 'close_time'

    # transfer the data type to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df


#%% Compute ADX
def compute_adx(df, period):
    """
    Input: df should include ['high','low','close']; period is the time interval of ADX
    """
    # 1. True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift()).abs()
    df['tr3'] = (df['low'] - df['close'].shift()).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # 2. +DM, -DM
    df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                         np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                         np.maximum(df['low'].shift() - df['low'], 0), 0)

    # 3. Wilder’s smoothing
    df['TR_smooth'] = df['TR'].rolling(period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(period).sum()

    # 4. DI
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    # 5. DX and ADX
    df['DX'] = 100 * ( (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI']) )
    df['ADX'] = df['DX'].rolling(period).mean()

    return df


#%% Compute RSI
def compute_rsi(data, period, ema=True):
    # Make sure there is enough data to calculate RSI
    if len(data) < period:
        print("Not enough data to calculate RSI")
        return pd.Series([np.nan] * len(data))

    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    if ema:
        # exponential moving average
        avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    else:
        # simple moving average
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


#%% Compute Bollinger Bands
def compute_bbands(data, window, nbdev=2):
    sma = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper = sma + nbdev * std  # upper line of Bollinger Bands
    lower = sma - nbdev * std  # lower line of Bollinger Bands

    return sma, upper, lower


#%% Compute MACD
def compute_macd(close, macd_span):
    # fast line and slow line
    ema_fast = close.ewm(span=macd_span[0], adjust=False).mean()
    ema_slow = close.ewm(span=macd_span[1], adjust=False).mean()

    # MACD line: ema_fast - ema_slow
    macd_line = ema_fast - ema_slow

    # signal line: EMA of MACD line
    signal_line = macd_line.ewm(span=macd_span[2], adjust=False).mean()

    # histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


#%% Safely concat
def safe_concat(df_initial, df_conc):  # concat 2 df without warning
    if df_initial.empty:  # if initial df is empty then use df_conc directly
        return df_conc
    elif not df_conc.empty:
        return pd.concat([df_initial, df_conc], ignore_index=True)


#%% RSI Only
def rsi_only(df, param):
    cash = 10000  # initial cash holding
    position = 0.0
    # Order Book, records the details of each transaction
    trades = pd.DataFrame(columns=['time', 'side', 'price', 'amount', 'present_price', 'PnL'])
    # Account Status, records the account status on an hourly basis
    account = pd.DataFrame(columns=['time', 'price', 'position', 'cash'])

    for index, row in tqdm(df.iterrows(), total=len(df)):
        pv = cash + position * row['close']
        if (row['RSI'] < 30) and (cash / pv > 0.4):
            allocation = min(pv * param['trade_ratio'], cash)  # ensure funds used no more than cash held
            buy_price = row['close'] * (1 + param['slippage'])
            amount = allocation * (1 - param['commision']) / buy_price

            cash -= buy_price * amount * (1 + param['commision'])
            position += amount
            new_trade = pd.DataFrame(
                {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                index=[len(trades)]
            )
            trades = safe_concat(trades, new_trade)

        if (row['RSI'] > 70) and (position > 0):
            allocation = position * row['close']
            sell_price = row['close'] * (1 - param['slippage'])
            amount = allocation * (1 - param['commision']) / sell_price
            amount = min(amount, position)  # ensure amount sold no more than positions held

            if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                cash += sell_price * amount
                position -= amount
                new_trade = pd.DataFrame(
                    {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount, 'stop': 0},
                    index=[len(trades)]
                )
                trades = safe_concat(trades, new_trade)

        # renew account status
        new_state = pd.DataFrame(
            {'time': row['close_time'], 'price': row['close'], 'position': position, 'cash': cash},
            index=[len(account)]
        )
        account = safe_concat(account, new_state)

    return trades, account


#%% RSI_BBands Strategy
def rsi_bbands(df, param):
    cash = 10000  # initial cash holding
    position = 0.0
    # Order Book, records the details of each transaction
    trades = pd.DataFrame(columns=['time', 'side', 'price', 'amount', 'present_price', 'PnL'])
    # Account Status, records the account status on an hourly basis
    account = pd.DataFrame(columns=['time', 'price', 'position', 'cash'])

    for index, row in tqdm(df.iterrows(), total=len(df)):
        pv = cash + position * row['close']
        # determine whether to buy in
        if cash / pv > 0.4:
            if (row['RSI'] < 30) and (row['close'] < row['BB_low']):  # RSI indicates oversold and the price falls below the lower bbands
                allocation = pv * param['trade_ratio']
                buy_price = row['close'] * (1 + param['slippage'])
                amount = allocation * (1 - param['commision']) / buy_price

                cash -= buy_price * amount * (1 + param['commision'])
                position += amount
                new_trade = pd.DataFrame(
                    {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                    index=[len(trades)]
                )
                trades = safe_concat(trades, new_trade)

        # determine whether to sell out
        if position > 0:
            if (row['RSI'] > 70) and (row['close'] > row['BB_up']):  # RSI denotes overbought and price rises above the upper bbands
                allocation = position * row['close']
                sell_price = row['close'] * (1 - param['slippage'])
                amount = allocation * (1 - param['commision']) / sell_price
                amount = min(amount, position)  # ensure amount sold no more than positions held

                if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                    cash += sell_price * amount
                    position -= amount
                    new_trade = pd.DataFrame(
                        {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount, 'stop': 0},
                        index=[len(trades)]
                    )
                    trades = safe_concat(trades, new_trade)

        # renew account status
        new_state = pd.DataFrame(
            {'time': row['close_time'], 'price': row['close'], 'position': position, 'cash': cash},
            index=[len(account)]
        )
        account = safe_concat(account, new_state)

    return trades, account


#%% RSI_BBands_MACD Strategy
def rsi_bbands_macd(df, param):
    cash = param['capital']  # initial cash holding
    position = param['position']
    # Order Book, records the details of each transaction
    trades = pd.DataFrame(columns=['time', 'side', 'price', 'amount', 'present_price', 'PnL'])
    # Account Status, records the account status on an hourly basis
    account = pd.DataFrame(columns=['time', 'price', 'position', 'cash'])

    for index, row in tqdm(df.iterrows(), total=len(df)):
        pv = cash + position * row['close']
        # Volatile Market Strategy: RSI + Bollinger Bands
        if row['ADX'] < param['adx_threshold'][0]:
            # determine whether to buy in
            if ((row['RSI'] < param['rsi_threshold'][0]) and (row['close'] < param['bb_threshold'][1] * row['BB_low'])
                    and (cash / pv > 0.4)):  # RSI indicates oversold and the price falls below the lower bbands
                allocation = min(pv * param['trade_ratio'], cash)  # ensure funds used no more than cash held
                buy_price = row['close'] * (1 + param['slippage'])
                amount = allocation * (1 - param['commision']) / buy_price

                cash -= buy_price * amount * (1 + param['commision'])
                position += amount
                new_trade = pd.DataFrame(
                    {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                    index=[len(trades)]
                )
                trades = safe_concat(trades, new_trade)

            # determine whether to sell out
            if ((row['RSI'] > param['rsi_threshold'][1]) and (row['close'] > param['bb_threshold'][0] * row['BB_up'])
                    and (position > 0)):  # RSI denotes overbought and price rises above the upper bbands
                allocation = position * row['close']
                sell_price = row['close'] * (1 - param['slippage'])
                amount = allocation * (1 - param['commision']) / sell_price
                amount = min(amount, position)  # ensure amount sold no more than positions held

                if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                    cash += sell_price * amount
                    position -= amount
                    new_trade = pd.DataFrame(
                        {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount, 'stop': 0},
                        index=[len(trades)]
                    )
                    trades = safe_concat(trades, new_trade)

        # Trending Market Strategy: RSI + MACD
        if row['ADX'] > param['adx_threshold'][1]:
            if row['MACD'] > row['signal']:  # uptrend
                if row['RSI'] < param['rsi_threshold'][0]:
                    allocation = min(pv * param['trade_ratio'], cash)  # ensure funds used no more than cash held
                    buy_price = row['close'] * (1 + param['slippage'])
                    amount = allocation * (1 - param['commision']) / buy_price

                    cash -= buy_price * amount * (1 + param['commision'])
                    position += amount
                    new_trade = pd.DataFrame(
                        {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                        index=[len(trades)]
                    )
                    trades = safe_concat(trades, new_trade)
                elif row['RSI'] > param['rsi_threshold'][1]:
                    allocation = position * row['close']
                    sell_price = row['close'] * (1 - param['slippage'])
                    amount = allocation * (1 - param['commision']) / sell_price
                    amount = min(amount, position)  # ensure amount sold no more than positions held

                    if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                        cash += sell_price * amount
                        position -= amount
                        new_trade = pd.DataFrame(
                            {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount, 'stop': 0},
                            index=[len(trades)]
                        )
                        trades = safe_concat(trades, new_trade)

            elif row['MACD'] < row['signal']:  # downtrend
                if row['RSI'] < param['rsi_threshold'][0]:
                    allocation = min(pv * param['trade_ratio'], cash)  # ensure funds used no more than cash held
                    buy_price = row['close'] * (1 + param['slippage'])
                    amount = allocation * (1 - param['commision']) / buy_price

                    cash -= buy_price * amount * (1 + param['commision'])
                    position += amount
                    new_trade = pd.DataFrame(
                        {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                        index=[len(trades)]
                    )
                    trades = safe_concat(trades, new_trade)
                elif row['RSI'] > param['rsi_threshold'][1]:
                    allocation = position * row['close']
                    sell_price = row['close'] * (1 - param['slippage'])
                    amount = allocation * (1 - param['commision']) / sell_price
                    amount = min(amount, position)  # ensure amount sold no more than positions held

                    if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                        cash += sell_price * amount
                        position -= amount
                        new_trade = pd.DataFrame(
                            {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount, 'stop': 0},
                            index=[len(trades)]
                        )
                        trades = safe_concat(trades, new_trade)

        # renew account status
        new_state = pd.DataFrame(
            {'time': row['close_time'], 'price': row['close'], 'position': position, 'cash': cash},
            index=[len(account)]
        )
        account = safe_concat(account, new_state)

    return trades, account


#%% Main Strategy
def strategy(df, param):
    cash = param['capital']  # initial cash holding
    position = param['position']
    # Order Book, records the details of each transaction
    trades = pd.DataFrame(columns=['time', 'side', 'price', 'amount', 'present_price', 'PnL'])
    # Account Status, records the account status on an hourly basis
    account = pd.DataFrame(columns=['time', 'price', 'position', 'cash'])

    for index, row in tqdm(df.iterrows(), total=len(df)):
        pv = cash + position * row['close']
        # Volatile Market Strategy: RSI + Bollinger Bands
        if row['ADX'] < param['adx_threshold'][0]:  # volatile market
            # determine whether to buy in
            if ((row['RSI'] < param['rsi_threshold'][0]) and (row['close'] < param['bb_threshold'][0] * row['BB_low'])
                    and (cash / pv > 0.4)):  # RSI indicates oversold and the price falls below the lower bbands
                allocation = min(pv * param['trade_ratio'], cash)  # ensure funds used no more than cash held
                buy_price = row['close'] * (1 + param['slippage'])
                amount = allocation * (1 - param['commision']) / buy_price

                cash -= buy_price * amount * (1 + param['commision'])
                position += amount
                new_trade = pd.DataFrame(
                    {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                    index=[len(trades)]
                )
                trades = safe_concat(trades, new_trade)

            # determine whether to sell out
            if ((row['RSI'] > param['rsi_threshold'][1]) and (row['close'] > param['bb_threshold'][1] * row['BB_up'])
                    and (position > 0)):  # RSI denotes overbought and price rises above the upper bbands
                allocation = position * row['close']
                sell_price = row['close'] * (1 - param['slippage'])
                amount = allocation * (1 - param['commision']) / sell_price
                amount = min(amount, position)  # ensure amount sold no more than positions held

                if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                    cash += sell_price * amount
                    position -= amount
                    new_trade = pd.DataFrame(
                        {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount, 'stop': 0},
                        index=[len(trades)]
                    )
                    trades = safe_concat(trades, new_trade)

        # Trending Market Strategy: RSI + MACD
        if row['ADX'] > param['adx_threshold'][1]:  # trending market
            if row['MACD'] > row['signal']:  # uptrend
                if row['RSI'] < param['rsi_up'][0]:  # RSI is less than the oversold bound in uptrend
                    allocation = min(pv * param['trade_ratio'], cash)  # ensure funds used no more than cash held
                    buy_price = row['close'] * (1 + param['slippage'])
                    amount = allocation * (1 - param['commision']) / buy_price

                    cash -= buy_price * amount * (1 + param['commision'])
                    position += amount
                    new_trade = pd.DataFrame(
                        {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                        index=[len(trades)]
                    )
                    trades = safe_concat(trades, new_trade)
                elif row['RSI'] > param['rsi_up'][1]:  # RSI is more than the overbought bound in uptrend
                    allocation = position * row['close']
                    sell_price = row['close'] * (1 - param['slippage'])
                    amount = allocation * (1 - param['commision']) / sell_price
                    amount = min(amount, position)  # ensure amount sold no more than positions held

                    if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                        cash += sell_price * amount
                        position -= amount
                        new_trade = pd.DataFrame(
                            {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount,
                             'stop': 0},
                            index=[len(trades)]
                        )
                        trades = safe_concat(trades, new_trade)

            elif row['MACD'] < row['signal']:  # downtrend
                if row['RSI'] < param['rsi_down'][0]:  # RSI is less than the oversold bound in downtrend
                    allocation = min(pv * param['trade_ratio'], cash)  # ensure funds used no more than cash held
                    buy_price = row['close'] * (1 + param['slippage'])
                    amount = allocation * (1 - param['commision']) / buy_price

                    cash -= buy_price * amount * (1 + param['commision'])
                    position += amount
                    new_trade = pd.DataFrame(
                        {'time': row['close_time'], 'side': 'buy', 'price': buy_price, 'amount': amount, 'stop': 0},
                        index=[len(trades)]
                    )
                    trades = safe_concat(trades, new_trade)
                elif row['RSI'] > param['rsi_down'][1]:  # RSI is more than the overbought bound in downtrend
                    allocation = position * row['close']
                    sell_price = row['close'] * (1 - param['slippage'])
                    amount = allocation * (1 - param['commision']) / sell_price
                    amount = min(amount, position)  # ensure amount sold no more than positions held

                    if amount > 10e-5:  # avoid extremely small trading amount caused by floating-point precision issues
                        cash += sell_price * amount
                        position -= amount
                        new_trade = pd.DataFrame(
                            {'time': row['close_time'], 'side': 'sell', 'price': sell_price, 'amount': amount,
                             'stop': 0},
                            index=[len(trades)]
                        )
                        trades = safe_concat(trades, new_trade)

        # renew account status
        new_state = pd.DataFrame(
            {'time': row['close_time'], 'price': row['close'], 'position': position, 'cash': cash},
            index=[len(account)]
        )
        account = safe_concat(account, new_state)

    return trades, account


#%% Performance Assessment
def performance_assessment(account, freq):
    account['cap'] = account['price'] * account['position'] + account['cash']

    # adjust the structure according to the data frequency
    if freq == 'daily':
        account = account.iloc[::24, :].copy()  # take one data point per day(24h)
        annual_freq = 365
    elif freq == 'hourly':
        account = account  # hourly data
        annual_freq = 24 * 365
    else:
        # throw an error when dealing with unknown frequency
        raise ValueError(f"不支持的频率: {freq}")

    account[f'{freq}_return'] = account['cap'].pct_change(fill_method=None).fillna(0)
    account['NAV'] = (account[f'{freq}_return'] + 1).cumprod()
    account['drawdown'] = account['NAV'] / account['NAV'].cummax() - 1
    account['return'] = account['price'].pct_change(fill_method=None).fillna(0)
    account['baseline'] = (account['return'] + 1).cumprod()

    summary = pd.DataFrame({
        'annual_return': [(account.iloc[-1]['NAV']) ** (annual_freq / len(account)) - 1],
        'NAV': [account.iloc[-1]['NAV']],
        'max_drawdown': [account['drawdown'].min()],
        'sharpe_ratio': [(account[f'{freq}_return'].mean() / account[f'{freq}_return'].std()) * np.sqrt(annual_freq)]
    })

    return account, summary


#%% Plot ADX
def plot_adx(df, param):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel('time')
    ax.plot(df['close_time'], df['ADX'], color='black', linewidth=1.5, label='ADX')
    ax.axhline(y=param[0], color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=param[1], color='green', linestyle='--', alpha=0.5)

    ax.set_ylabel('ADX', fontsize=12, fontweight='bold', color='black')
    ax.tick_params(axis='y', labelcolor='black')

    plt.show()


#%% Plot RSI
def plot_rsi(df, param):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel('time')
    ax.plot(df['close_time'], df['RSI'], color='black', linewidth=1.5, label='RSI')
    ax.axhline(y=param['rsi_threshold'][0], color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=param['rsi_threshold'][1], color='green', linestyle='--', alpha=0.5)

    ax.set_ylabel('RSI', fontsize=12, fontweight='bold', color='black')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelcolor='black')

    plt.show()


#%% Plot Bollinger Bands
def plot_bbands(df, param):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel('time')
    ax.plot(df['close_time'], df['close'], color='black', linewidth=1.5, label='sma')
    ax.plot(df['close_time'], param['bb_threshold'][1] * df['BB_up'],
            color='green', linestyle='--', alpha=0.5, label='up')
    ax.plot(df['close_time'], param['bb_threshold'][0] * df['BB_low'],
            color='red', linestyle='--', alpha=0.5, label='down')

    ax.tick_params(axis='y', labelcolor='black')

    plt.show()


#%% Plot MACD
def plot_macd(df):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel('time')
    ax.plot(df['close_time'], df['MACD'], color='blue', linewidth=1.5, label='sma')
    ax.plot(df['close_time'], df['signal'], color='green', linestyle='--', alpha=0.5, label='up')
    ax.tick_params(axis='y', labelcolor='black')

    plt.show()


#%% Plot NAV and Baseline
def plot_nav(account):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel('time')
    # draw nav line and baseline
    ax.plot(account['time'], account['NAV'], color='red', label='NAV')
    ax.plot(account['time'], account['baseline'], color='blue', label='baseline')
    # show legend
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


#%% Mark trading points
def mark_trading(account, trades):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel('time')
    # draw nav line and baseline
    ax.plot(account['time'], account['NAV'], color='red', label='NAV')
    ax.plot(account['time'], account['baseline'], color='blue', label='baseline')
    # mark trading points
    for _, t in trades.iterrows():
        if t['side'] == 'buy':  # mark buy points
            # green dashed line for buy entries
            ax.axvline(x=t['time'], color='green', linestyle='--', alpha=0.7,
                       label='BUY' if 'BUY' not in ax.get_legend_handles_labels()[1] else "")
        elif t['side'] == 'sell':  # mark sell points
            # orange dashed line for sell exits
            ax.axvline(x=t['time'], color='orange', linestyle='--', alpha=0.7,
                       label='SELL' if 'SELL' not in ax.get_legend_handles_labels()[1] else "")

    # show legend
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
