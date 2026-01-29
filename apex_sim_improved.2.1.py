import pandas as pd
from datetime import datetime, timedelta

# ===== CONFIGURATION - ADJUST THESE VALUES =====
EVAL_COST = 40
PA_COST = 105
EVAL_TARGET = 1500  # Profit target to pass eval
EVAL_TRAILING_DD = 1500  # Trailing drawdown during eval
EVAL_TIMEOUT_DAYS = 30  # Eval expires after 30 days
PA_MIN_DAYS = 8
PA_MIN_WINS = 5  # Need 5 winning trades for payout
PA_PAYOUT_MIN = 1600  # Minimum PnL to request payout
PA_TRAILING_DD = 1500
PA_ACCOUNT_BALANCE = 25000  # Actual account balance
SAFETY_NET_BALANCE = 26600  # Account balance where DD stops trailing (25k + 1.6k)
SAFETY_NET_FLOOR = 25100  # Fixed DD threshold once above safety net
BASE_PAYOUT = 500  # Base payout at 1.6k profit
MAX_PAYOUT = 1500  # Maximum single payout amount

# Load your backtest CSV
df = pd.read_csv("backtest_results_6pmClose_730pmClose_Testing.csv")

# Normalize column names to lowercase
df.columns = df.columns.str.lower()

# Filter out NO_FILL trades (only keep completed trades)
df = df[df["status"] != "NO_FILL"].copy()

# Map exit types to your specified PnL values
mapping = {"TP": 704, "SL": -420} 
df["pnl"] = df["exit_type"].map(mapping)

# Convert date column to datetime and sort
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Loaded {len(df)} completed trades (excluding NO_FILL)")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Win Rate: {(df['exit_type'] == 'TP').sum() / len(df) * 100:.1f}%")
print(f"\nStarting simulation with:")
print(f"  Eval Cost: ${EVAL_COST}")
print(f"  PA Cost: ${PA_COST}")
print(f"  Eval Timeout: {EVAL_TIMEOUT_DAYS} days")
print(f"  Win per trade: ${mapping['TP']}")
print(f"  Loss per trade: ${mapping['SL']}")
print("=" * 60)

# ===== SIMULATION STATE =====
state = "eval"
bankroll = -EVAL_COST
current_pnl = 0.0  # PnL relative to starting balance
daily_pnl = 0.0
previous_date = None
pa_dates = set()
pa_wins = 0  # Count winning trades
eval_peak = 0
pa_peak = 0
payout_count = 0
eval_start_date = df["date"].iloc[0]  # Track when eval started

# Results
rows = []

for idx, row in df.iterrows():
    date = row["date"]
    pnl = row["pnl"]
    tag = ""
    failed = False
    eval_passed_this_trade = False
    
    # Detect new trading day
    if date != previous_date:
        daily_pnl = 0
        previous_date = date
    
    daily_pnl += pnl
    
    # ===== EVALUATION PHASE =====
    if state == "eval":
        # Check if eval has expired (30 days)
        days_in_eval = (date - eval_start_date).days
        if days_in_eval > EVAL_TIMEOUT_DAYS:
            tag = "eval_expired_30days"
            bankroll -= EVAL_COST  # Buy new eval
            current_pnl = 0
            eval_peak = 0
            eval_start_date = date
            failed = True
        else:
            current_pnl += pnl
            eval_peak = max(eval_peak, current_pnl)
            tag = "eval_trade"
            
            # Check trailing drawdown (blown at peak - 1500)
            trailing_threshold = eval_peak - EVAL_TRAILING_DD
            if current_pnl < trailing_threshold:
                tag = "eval_failed_trailing_dd"
                state = "failed"
                bankroll -= EVAL_COST  # Pay for new eval
                current_pnl = 0
                eval_peak = 0
                daily_pnl = 0
                eval_start_date = date
                state = "eval"
                failed = True
            
            # Check if passed eval (hit target)
            elif current_pnl >= EVAL_TARGET:
                tag = "passed_eval"
                bankroll -= PA_COST  # Pay for PA account
                state = "pa"
                eval_passed_this_trade = True
                # Don't process PA logic this trade - it was an eval trade
                # Reset for PA starting next trade
                current_pnl = 0
                pa_peak = 0
                pa_dates = set()
                pa_wins = 0
    
    # ===== PERFORMANCE ACCOUNT PHASE =====
    elif state == "pa":
        current_pnl += pnl
        pa_peak = max(pa_peak, current_pnl)
        
        # Add current date to PA dates and count wins
        pa_dates.add(date)
        if pnl > 0:
            pa_wins += 1
        
        tag = "pa_trade"
        
        # Calculate actual account balance (25k base + PnL)
        actual_balance = PA_ACCOUNT_BALANCE + current_pnl
        
        # Determine trailing drawdown threshold
        # If peak balance ever reached >= 26600, DD locks at 25100 (PnL of $100)
        peak_balance = PA_ACCOUNT_BALANCE + pa_peak
        
        if peak_balance >= SAFETY_NET_BALANCE:
            # DD is locked - cannot go below $25,100 balance = $100 PnL
            trailing_threshold = SAFETY_NET_FLOOR - PA_ACCOUNT_BALANCE  # 25100 - 25000 = 100
        else:
            # Normal trailing: peak PnL - $1,500
            trailing_threshold = pa_peak - PA_TRAILING_DD
        
        # Check if account failed (PnL drops below threshold)
        if current_pnl < trailing_threshold:
            tag = f"pa_failed_trailing_dd (PnL: ${current_pnl:.0f}, threshold: ${trailing_threshold:.0f})"
            failed = True
            state = "failed"
            bankroll -= EVAL_COST  # Pay for new eval
            current_pnl = 0
            eval_peak = 0
            pa_peak = 0
            daily_pnl = 0
            pa_dates = set()
            pa_wins = 0
            payout_count = 0
            eval_start_date = date
            state = "eval"
        
        # Check payout eligibility (8+ days, 5+ wins, 1600+ PnL)
        elif (len(pa_dates) >= PA_MIN_DAYS and 
              pa_wins >= PA_MIN_WINS and
              current_pnl >= PA_PAYOUT_MIN):
            
            payout_count += 1
            
            # Calculate payout amount
            # Base $500 at 1.6k + anything above 1.6k, capped at $1500 total
            profit_above_min = current_pnl - PA_PAYOUT_MIN
            payout_amount = min(BASE_PAYOUT + profit_above_min, MAX_PAYOUT)
            
            bankroll += payout_amount
            tag = f"payout_{payout_count}_${payout_amount:.0f}"
            
            # After payout, reduce PnL and reset tracking
            current_pnl -= payout_amount
            pa_dates = set()
            pa_wins = 0
            # Note: pa_peak does NOT reset - keeps trailing from historical high
            # unless we're now above safety net, then it doesn't matter
    
    # Log row state
    rows.append({
        "date": date,
        "pnl": pnl,
        "daily_pnl": daily_pnl,
        "state": state if not failed else "resetting",
        "acct_pnl": current_pnl,
        "actual_balance": PA_ACCOUNT_BALANCE + current_pnl if state == "pa" else current_pnl,
        "bankroll": bankroll,
        "pa_days": len(pa_dates) if state == "pa" else 0,
        "pa_wins": pa_wins if state == "pa" else 0,
        "payout_num": payout_count,
        "eval_days": (date - eval_start_date).days if state == "eval" else 0,
        "tag": tag
    })

# Build output
out = pd.DataFrame(rows)

# Save results
out.to_csv("simulated_apex_results.csv", index=False)

# ===== SUMMARY STATS =====
print("\n" + "=" * 60)
print("APEX TRADER FUNDING SIMULATION RESULTS")
print("=" * 60)
print(f"\nFinal Bankroll: ${bankroll:.2f}")
print(f"Final State: {state}")
if state == "pa":
    print(f"Account PnL: ${current_pnl:.2f}")
    print(f"Actual Balance: ${PA_ACCOUNT_BALANCE + current_pnl:.2f}")
else:
    print(f"Eval PnL: ${current_pnl:.2f}")

payouts = out[out["tag"].str.contains("payout", na=False)]
print(f"\nTotal Payouts: {len(payouts)}")
if len(payouts) > 0:
    total_payout = payouts["tag"].str.extract(r'\$(\d+)')[0].astype(float).sum()
    print(f"Total Payout Amount: ${total_payout:.2f}")

eval_fails = out[out["tag"].str.contains("eval_failed", na=False)]
eval_expires = out[out["tag"].str.contains("eval_expired", na=False)]
print(f"\nEval Failures (DD): {len(eval_fails)}")
print(f"Eval Failures (30-day timeout): {len(eval_expires)}")

pa_fails = out[out["tag"].str.contains("pa_failed", na=False)]
print(f"PA Failures: {len(pa_fails)}")

eval_passes = out[out["tag"].str.contains("passed_eval", na=False)]
print(f"\nEvals Passed: {len(eval_passes)}")

total_eval_attempts = len(eval_passes) + len(eval_fails) + len(eval_expires)
if total_eval_attempts > 0:
    pass_rate = len(eval_passes) / total_eval_attempts * 100
    print(f"Eval Pass Rate: {pass_rate:.1f}%")

total_fees = (EVAL_COST * (total_eval_attempts)) + (PA_COST * len(eval_passes))
print(f"\nTotal Fees Paid: ${total_fees:.2f}")
print(f"  Eval fees: ${EVAL_COST * total_eval_attempts:.2f}")
print(f"  PA fees: ${PA_COST * len(eval_passes):.2f}")

print(f"\nNet Profit: ${bankroll + EVAL_COST:.2f}")  # Add back initial eval cost
print(f"Total Trades Simulated: {len(out)}")

print(f"\nCONFIGURATION USED:")
print(f"  Eval Cost: ${EVAL_COST}")
print(f"  PA Cost: ${PA_COST}")
print(f"  Eval Target: ${EVAL_TARGET}")
print(f"  Eval Timeout: {EVAL_TIMEOUT_DAYS} days")
print(f"  PA Account Balance: ${PA_ACCOUNT_BALANCE}")
print(f"  Safety Net Balance: ${SAFETY_NET_BALANCE}")
print(f"  Base Payout: ${BASE_PAYOUT}")
print(f"  Max Payout: ${MAX_PAYOUT}")

print("\nSaved detailed results to: simulated_apex_results.csv")
print("=" * 60)

# Preview key events
print("\nKEY EVENTS:")
key_events = out[out["tag"].str.contains("passed|payout|failed|expired", na=False)]
if len(key_events) > 0:
    display_cols = ["date", "pnl", "acct_pnl", "actual_balance", "bankroll", "pa_wins", "tag"]
    print(key_events[display_cols].to_string(index=False))
else:
    print("No key events (passes, payouts, or failures) occurred.")