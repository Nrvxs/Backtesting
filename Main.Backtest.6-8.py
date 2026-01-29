import pandas as pd
import numpy as np
from datetime import time

# Load your CSV file
df = pd.read_excel('C:\\Users\\natha\\OneDrive\\Desktop\\Main\\Work\\APEX\\Backtesting Data NASDAQ\\Good Data\\Specific Case\\Year_2016-2025.xlsx')

# Keep only first 7 columns and remove empty rows
df = df.iloc[:, :7]
df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
df = df.dropna()

# Convert Date to datetime (M/D/Y format)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# Add day of week
df['DayOfWeek'] = df['Date'].dt.day_name()

# Filter for valid trading days (Monday-Thursday)
df['ValidTradingDay'] = df['DayOfWeek'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])

# Calculate EMAs
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# Extract time as time object for comparison
df['TimeOnly'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

# Identify 6PM and 7:30PM bars 
df['Is6PM'] = (df['TimeOnly'] == time(18, 0)) & df['ValidTradingDay']
df['Is730PM'] = (df['TimeOnly'] == time(19, 30)) & df['ValidTradingDay']

# Create a trading session ID for grouping bars by evening
df['TradingSession'] = df['Date'].dt.date

# Initialize results
trades = []

# Process each trading session
for session_id in df[df['Is6PM']]['TradingSession'].unique():
    session_data = df[df['TradingSession'] == session_id].copy()
    
    # Get 6PM-7:30PM range (18:00 to 19:30 in data = 6PM-7:30PM EST)
    pm_6_to_730 = session_data[(session_data['TimeOnly'] >= time(18, 0)) & 
                                (session_data['TimeOnly'] <= time(19, 30))]
    
    if len(pm_6_to_730) < 3:  # Need at least 6PM, 6:30PM, 7:30PM bars
        continue
    
    # Get 6PM bar (first bar in range) - must be exactly 18:00
    bar_6pm = pm_6_to_730[pm_6_to_730['TimeOnly'] == time(18, 0)]
    if len(bar_6pm) == 0:
        continue
    
    # Get 7:30PM bar - must be exactly 19:30
    bar_730pm = pm_6_to_730[pm_6_to_730['TimeOnly'] == time(19, 30)]
    if len(bar_730pm) == 0:
        continue
    
    # Get key prices from specific bars
    open_6pm = bar_6pm.iloc[0]['Open']
    close_730pm = bar_730pm.iloc[0]['Close']
    ema_20_730pm = bar_730pm.iloc[0]['EMA_20']
    ema_50_730pm = bar_730pm.iloc[0]['EMA_50']
    
    trade_date = bar_6pm.iloc[0]['Date']  
    """
    # debug BEFORE calculating range
    print(f"\nDEBUG: Bars in 6PM-7:30PM range for {trade_date.date()}:")
    print(f"Session ID: {session_id}, Total bars in session_data: {len(session_data)}")
    print(f"Bars matching time filter: {len(pm_6_to_730)}")
    for idx, row in pm_6_to_730.iterrows():
        print(f"  Date: {row['Date'].date()}, Time: {row['TimeOnly']}, High: {row['High']}, Low: {row['Low']}")"""
    
    # calculate range
    range_high = pm_6_to_730['High'].max()
    range_low = pm_6_to_730['Low'].min()
    """
    print(f"Calculated Range High: {range_high}, Range Low: {range_low}\n")"""
    
    # Determine setup type
    setup_type = None
    entry_price = None
    """
    print(f"\n--- DEBUG for {trade_date.date()} ---")
    print(f"6PM Open: {open_6pm}")
    print(f"7:30PM Close: {close_730pm}")
    print(f"EMA_20 at 7:30PM: {ema_20_730pm}")
    print(f"EMA_50 at 7:30PM: {ema_50_730pm}")
    print(f"Range High: {range_high}")
    print(f"Range Low: {range_low}")
    print(f"LONG conditions: Close > Open? {close_730pm > open_6pm}, Close > EMA20? {close_730pm > ema_20_730pm}, Close > EMA50? {close_730pm > ema_50_730pm}")
    print(f"SHORT conditions: Close < Open? {close_730pm < open_6pm}, Close < EMA20? {close_730pm < ema_20_730pm}, Close < EMA50? {close_730pm < ema_50_730pm}")
"""
    # LONG setup
    if close_730pm > open_6pm and close_730pm > ema_20_730pm and close_730pm > ema_50_730pm:
        setup_type = 'LONG'
        entry_price = range_high
    
    # SHORT setup
    elif close_730pm < open_6pm and close_730pm < ema_20_730pm and close_730pm < ema_50_730pm:
        setup_type = 'SHORT'
        entry_price = range_low
    
    if setup_type is None:
        continue
    original_entry_price = float(entry_price)
    
    # Check for entry after 7:30PM until 10:00PM EST 
    bars_after_730pm = session_data[(session_data['TimeOnly'] > time(19, 30)) & 
                                     (session_data['TimeOnly'] <= time(22, 30)) &
                                     (session_data['ValidTradingDay'])]
    
    entry_filled = False
    entry_bar_idx = None
    
    for idx, bar in bars_after_730pm.iterrows():
        if setup_type == 'LONG' and bar['High'] >= original_entry_price:
            entry_filled = True
            entry_bar_idx = idx
            break
        elif setup_type == 'SHORT' and bar['Low'] <= original_entry_price:
            entry_filled = True
            entry_bar_idx = idx
            break
    
    if not entry_filled:
        trades.append({
            'Date': trade_date,
            'Setup': setup_type,
            'Entry_Price': original_entry_price,
            'Status': 'NO_FILL',
            'Exit_Price': None,
            'Points_PL': 0,
            'Dollar_PL': 0
        })
        continue
    
    # Trade is filled, now track exit
    if setup_type == 'LONG':
        tp_level = original_entry_price + 51
        sl_level = original_entry_price - 30
    else:  # SHORT
        tp_level = original_entry_price - 51
        sl_level = original_entry_price + 30
    
    # Check bars after entry for TP or SL
    bars_after_entry = df.loc[entry_bar_idx+1:]
    
    exit_price = None
    exit_type = None
    
    for idx, bar in bars_after_entry.iterrows():
        if setup_type == 'LONG':
            if bar['High'] >= tp_level:
                exit_price = tp_level
                exit_type = 'TP'
                break
            elif bar['Low'] <= sl_level:
                exit_price = sl_level
                exit_type = 'SL'
                break
        else:  # SHORT
            if bar['Low'] <= tp_level:
                exit_price = tp_level
                exit_type = 'TP'
                break
            elif bar['High'] >= sl_level:
                exit_price = sl_level
                exit_type = 'SL'
                break
    
    # Calculate P&L
    if exit_price:
        if setup_type == 'LONG':
            points_pl = exit_price - original_entry_price
        else:
            points_pl = original_entry_price - exit_price
        
        dollar_pl = points_pl * 1 * 0.5  # 1 contracts * $0.50 per point (MNQ)
        
        trades.append({
            'Date': trade_date,
            'Setup': setup_type,
            'Entry_Price': original_entry_price,
            'Exit_Price': exit_price,
            'Exit_Type': exit_type,
            'Points_PL': points_pl,
            'Dollar_PL': dollar_pl,
            'Status': 'COMPLETED'
        })

# Create results DataFrame
# Create results DataFrame
results_df = pd.DataFrame(trades)

# Calculate metrics
if len(results_df) > 0:
    completed_trades = results_df[results_df['Status'] == 'COMPLETED']
    
    if len(completed_trades) > 0:
        total_trades = len(completed_trades)
        winning_trades = len(completed_trades[completed_trades['Dollar_PL'] > 0])
        losing_trades = len(completed_trades[completed_trades['Dollar_PL'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pl = completed_trades['Dollar_PL'].sum()
        avg_win = completed_trades[completed_trades['Dollar_PL'] > 0]['Dollar_PL'].mean() if winning_trades > 0 else 0
        avg_loss = completed_trades[completed_trades['Dollar_PL'] < 0]['Dollar_PL'].mean() if losing_trades > 0 else 0
        
        total_wins = completed_trades[completed_trades['Dollar_PL'] > 0]['Dollar_PL'].sum()
        total_losses = abs(completed_trades[completed_trades['Dollar_PL'] < 0]['Dollar_PL'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # ADDITIONAL METRICS
        
        # 1. Calculate returns as percentage of capital
        initial_capital = 1500  # Adjust this to your actual account size
        completed_trades['Return_Pct'] = (completed_trades['Dollar_PL'] / initial_capital) * 100
        
        # 2. Calculate drawdown
        completed_trades_sorted = completed_trades.sort_values('Date')
        completed_trades_sorted['Cumulative_PL'] = completed_trades_sorted['Dollar_PL'].cumsum()
        completed_trades_sorted['Running_Max'] = completed_trades_sorted['Cumulative_PL'].cummax()
        completed_trades_sorted['Drawdown'] = completed_trades_sorted['Cumulative_PL'] - completed_trades_sorted['Running_Max']
        
        max_drawdown = completed_trades_sorted['Drawdown'].min()
        max_drawdown_pct = (max_drawdown / initial_capital) * 100
        
        # 3. Calculate Sharpe Ratio
        returns = completed_trades['Dollar_PL'].values
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Risk-free rate (adjust based on current rates)
        annual_rf_rate = 0.0425  # 4.5% annual
        
        # Calculate actual trading frequency
        date_range = (completed_trades['Date'].max() - completed_trades['Date'].min()).days
        trades_per_day = total_trades / date_range if date_range > 0 else 0
        trades_per_year = trades_per_day * 252
        
        # Risk-free return per trade
        rf_per_trade = (annual_rf_rate / trades_per_year) if trades_per_year > 0 else 0
        rf_dollar_per_trade = rf_per_trade * initial_capital
        
        if std_return > 0 and trades_per_year > 0:
            sharpe_ratio = ((avg_return - rf_dollar_per_trade) / std_return) * np.sqrt(trades_per_year)
        else:
            sharpe_ratio = 0
        
        annualized_return = avg_return * trades_per_year if trades_per_year > 0 else 0
        annualized_volatility = std_return * np.sqrt(trades_per_year) if trades_per_year > 0 else 0
        
        # 4. Calculate Sortino Ratio (only penalizes downside volatility)
        downside_returns = completed_trades[completed_trades['Dollar_PL'] < 0]['Dollar_PL'].values
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns, ddof=1)
            if downside_std > 0 and trades_per_year > 0:
                sortino_ratio = ((avg_return - rf_dollar_per_trade) / downside_std) * np.sqrt(trades_per_year)
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
        
        # 5. Calculate Calmar Ratio (Return / Max Drawdown)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        # 6. Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        # 7. Risk of Ruin (simplified Kelly Criterion)
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        kelly_pct = (win_rate/100) - ((1 - win_rate/100) / win_loss_ratio) if win_loss_ratio > 0 else 0
        
        # 8. Average trade duration
        # Note: This requires tracking entry and exit times, placeholder for now
        
        print("=" * 60)
        print("COMPREHENSIVE TRADING STRATEGY BACKTEST RESULTS")
        print("=" * 60)
        print(f"\n📊 TRADE STATISTICS")
        print(f"Total Setups Found: {len(results_df)}")
        print(f"No Fill (Cancelled): {len(results_df[results_df['Status'] == 'NO_FILL'])}")
        print(f"Completed Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        
        print(f"\n💰 PROFIT & LOSS")
        print(f"Total P&L: ${total_pl:,.2f}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Largest Win: ${completed_trades['Dollar_PL'].max():,.2f}")
        print(f"Largest Loss: ${completed_trades['Dollar_PL'].min():,.2f}")
        print(f"Expectancy per Trade: ${expectancy:,.2f}")
        
        print(f"\n📈 RISK METRICS")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
        print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        print(f"\n🎯 RISK-ADJUSTED RETURNS")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print(f"Annualized Return: ${annualized_return:,.2f}")
        print(f"Annualized Volatility: ${annualized_volatility:,.2f}")
        
        print(f"\n💡 POSITION SIZING")
        print(f"Optimal Kelly %: {kelly_pct*100:.2f}%")
        print(f"Recommended Kelly (25% of optimal): {kelly_pct*25:.2f}%")
        
        print(f"\n📅 FREQUENCY")
        print(f"Trading Days Analyzed: {date_range}")
        print(f"Trades per Year (estimated): {trades_per_year:.1f}")
        
        print("=" * 60)
        
        # Save detailed results to CSV
        results_df.to_csv('backtest_results_6pmClose_730pmClose_Testing.csv', index=False)
        print("\nDetailed results saved to 'backtest_results_6pmClose_730pmClose_Testing.csv'")
    else:
        print("No completed trades found.")
else:
    print("No trading setups found in the data.")