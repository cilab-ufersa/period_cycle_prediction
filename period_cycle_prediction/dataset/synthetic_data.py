from period_cycle_prediction.utils import generate_synthetic_data

df = generate_synthetic_data(duration_cycle=5, start_day=25, year=2021, start_month_index=1, number_of_cycle = 5, period_duration = 30)

print(df.head())

