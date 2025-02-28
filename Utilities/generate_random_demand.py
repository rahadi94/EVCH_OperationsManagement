import pandas as pd
import numpy as np
import random

# Load the original dataset to get column structure
import pandas as pd
import numpy as np
import random

import pandas as pd
import numpy as np
import random

import pandas as pd
import numpy as np
import random

def generate_random_demand(sim_start_day):
    file_path = "EV_Charging_Clusters/Cache/requests_sample_2_size_limit_200_max_charge_rate_50.pkl"
    df = pd.read_pickle(file_path)

    # Set seed for reproducibility
    np.random.seed(42)

    # Number of rows to generate
    num_rows = 100

    # Identify column types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Generate synthetic data dictionary
    synthetic_data = {}

    # Generate numeric columns using a normal distribution
    for col in numeric_cols:
        mean = df[col].mean() if not df[col].isnull().all() else 10  # Default mean = 10 if all values are NaN
        std = df[col].std() if not df[col].isnull().all() else 5      # Default std = 5
        synthetic_data[col] = np.abs(np.random.normal(loc=mean, scale=std, size=num_rows))  # Ensure positive values

    # Generate categorical columns by randomly sampling from original unique values
    for col in categorical_cols:
        unique_values = df[col].dropna().unique().tolist()  # Get unique non-null values
        if unique_values:
            synthetic_data[col] = [random.choice(unique_values) for _ in range(num_rows)]
        else:
            synthetic_data[col] = ["Unknown"] * num_rows  # Default if no categories exist

    # Generate datetime columns with both Entry and Exit on the same day
    start_date = pd.Timestamp(sim_start_day)
    entry_times = start_date + pd.to_timedelta(np.random.uniform(0, 24, num_rows), unit="h")  # Random time on the same day

    # Ensure exit times remain within the same day
    midnight = start_date + pd.Timedelta(days=1)  # Next day's midnight
    max_possible_stay = (midnight - entry_times).total_seconds() / 3600  # Convert to hours
    hours_stay = np.minimum(np.random.exponential(scale=5, size=num_rows), max_possible_stay)  # Ensure exit is on the same day
    exit_times = entry_times + pd.to_timedelta(hours_stay, unit="h")

    # Create DataFrame for time-related data
    time_df = pd.DataFrame({"EntryDateTime": entry_times, "ExitDateTime": exit_times})

    # Calculate derived time-related columns
    time_df["HoursStay"] = (time_df["ExitDateTime"] - time_df["EntryDateTime"]).dt.total_seconds() / 3600
    time_df["MinutesStay"] = time_df["HoursStay"] * 60
    time_df["Year"] = time_df["EntryDateTime"].dt.year
    time_df["EntryDate"] = time_df["EntryDateTime"].dt.date
    time_df["ExitDate"] = time_df["ExitDateTime"].dt.date
    time_df["EntryHour"] = time_df["EntryDateTime"].dt.hour
    time_df["ExitHour"] = time_df["ExitDateTime"].dt.hour
    time_df["EntryDayOfWeek"] = time_df["EntryDateTime"].dt.weekday
    time_df["EntryWeekday_yn"] = time_df["EntryDayOfWeek"].apply(lambda x: 1 if x < 5 else 0)  # 1 = Weekday, 0 = Weekend
    time_df["EntryHoliday_yn"] = time_df["EntryWeekday_yn"]  # Placeholder for holidays

    # Round datetime to nearest 5 minutes
    time_df["EntryDateTime5min"] = time_df["EntryDateTime"].dt.round("5min")
    time_df["ExitDateTime5min"] = time_df["ExitDateTime"].dt.round("5min")

    # Combine synthetic categorical/numeric data with time-related data
    synthetic_df = pd.DataFrame(synthetic_data)
    final_df = pd.concat([synthetic_df, time_df], axis=1)

    return final_df