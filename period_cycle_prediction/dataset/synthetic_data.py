import sys
sys.path.append("../period_cycle_prediction/")
from period_cycle_prediction.utils import generate_synthetic_data

dataset = generate_synthetic_data(duration_cycle=5, start_day=25, year=2021, start_month_index=1, number_of_cycle = 24, period_duration = 30)

dataset.to_csv('period_cycle_prediction\dataset\synthetic_data.csv', index=False)
