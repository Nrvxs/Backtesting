import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('simulated_apex_results.csv')
df.columns = df.columns.str.strip()
df = df.sort_values('date').reset_index(drop=True)

print("="*70)
print("APEX EVAL ANALYSIS - CORRECTED")
print("="*70 + "\n")

# Count eval attempts and their outcomes
# An eval attempt starts and either: passes, expires, or fails DD
eval_expired = len(df[df['tag'] == 'eval_expired_30days'])
eval_failed_dd = len(df[df['tag'] == 'eval_failed_trailing_dd'])
eval_passed = len(df[df['tag'] == 'passed_eval'])

total_eval_attempts = eval_expired + eval_failed_dd + eval_passed

print(f"Total eval attempts: {total_eval_attempts}")
print(f"  Passed: {eval_passed}")
print(f"  Failed (30-day timeout): {eval_expired}")
print(f"  Failed (trailing DD): {eval_failed_dd}")
print("\n" + "="*70 + "\n")

# For each passed eval, track if they reached a payout
# Find all passed_eval rows
passed_indices = df[df['tag'] == 'passed_eval'].index.tolist()

# Track PA account outcomes
pa_accounts = []

for pass_idx in passed_indices:
    # Get all PA trades after this pass until next reset or end
    pa_data = []
    reached_payout = False
    max_payout = 0
    days_to_first_payout = None
    failed = False
    
    # Look at rows after passing eval
    for i in range(pass_idx + 1, len(df)):
        row = df.iloc[i]
        tag = row['tag']
        
        # Check if this PA account failed
        if 'pa_failed' in tag:
            failed = True
            break
        
        # Check if we hit a new eval attempt (means this PA ended)
        if tag in ['eval_expired_30days', 'eval_failed_trailing_dd', 'passed_eval']:
            break
        
        # Check for payouts
        if 'payout' in tag:
            if not reached_payout:
                reached_payout = True
                # Days from passing eval to first payout
                days_to_first_payout = i - pass_idx
            
            # Extract payout number
            payout_num = int(tag.split('_')[1])
            max_payout = max(max_payout, payout_num)
    
    pa_accounts.append({
        'passed_eval': True,
        'reached_payout': reached_payout,
        'max_payout': max_payout,
        'days_to_first_payout': days_to_first_payout,
        'failed': failed
    })

pa_df = pd.DataFrame(pa_accounts)

# QUESTION 1: What % of passed evals reach at least 1 payout?
passed_with_payout = pa_df[pa_df['reached_payout'] == True]
payout_percentage = (len(passed_with_payout) / len(pa_df) * 100) if len(pa_df) > 0 else 0

print("QUESTION 1: Payout Success Rate After Passing Eval")
print(f"Accounts that passed eval: {len(pa_df)}")
print(f"Accounts that reached ≥1 payout: {len(passed_with_payout)}")
print(f"Percentage: {payout_percentage:.2f}%")
print("\n" + "="*70 + "\n")

# QUESTION 2: Average payouts per account (when at least 1 reached)
if len(passed_with_payout) > 0:
    avg_payouts = passed_with_payout['max_payout'].mean()
    total_payouts = passed_with_payout['max_payout'].sum()
    
    print("QUESTION 2: Average Payouts Per Account (When ≥1 Reached)")
    print(f"Accounts with at least 1 payout: {len(passed_with_payout)}")
    print(f"Average number of payouts: {avg_payouts:.2f}")
    print(f"Min payouts: {passed_with_payout['max_payout'].min():.0f}")
    print(f"Max payouts: {passed_with_payout['max_payout'].max():.0f}")
    print(f"Median payouts: {passed_with_payout['max_payout'].median():.2f}")
    print(f"Total payouts across all accounts: {total_payouts:.0f}")
else:
    print("QUESTION 2: No accounts reached any payouts")
    
print("\n" + "="*70 + "\n")

# QUESTION 3: Average days from eval start to first payout
with_timing = passed_with_payout[passed_with_payout['days_to_first_payout'].notna()]

if len(with_timing) > 0:
    avg_days = with_timing['days_to_first_payout'].mean()
    
    print("QUESTION 3: Time from Passing Eval to First Payout")
    print(f"Accounts with timing data: {len(with_timing)}")
    print(f"Average days (rows) to first payout: {avg_days:.1f} days")
    print(f"Min days: {with_timing['days_to_first_payout'].min():.0f}")
    print(f"Max days: {with_timing['days_to_first_payout'].max():.0f}")
    print(f"Median days: {with_timing['days_to_first_payout'].median():.1f}")
else:
    print("QUESTION 3: No payout timing data available")
    
print("\n" + "="*70 + "\n")

# Additional insights
if len(passed_with_payout) > 0:
    print("PAYOUT DISTRIBUTION:")
    payout_dist = passed_with_payout['max_payout'].value_counts().sort_index()
    for payouts, count in payout_dist.items():
        percentage = (count / len(passed_with_payout) * 100)
        print(f"  {int(payouts)} payout(s): {count} accounts ({percentage:.1f}%)")

print(f"\nEval pass rate: {(eval_passed/total_eval_attempts)*100:.2f}%")

# Show breakdown
print(f"\nPA Account Outcomes:")
print(f"  Reached payout: {len(passed_with_payout)}")
print(f"  Never reached payout: {len(pa_df) - len(passed_with_payout)}")
if len(pa_df) > 0:
    print(f"  Failed PA accounts: {pa_df['failed'].sum()}")