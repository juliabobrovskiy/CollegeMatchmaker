from urllib.request import urlopen
from json import loads
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

YEAR = 2014
def data_to_pandas(url):
    response = urlopen(url)
    data = loads(response.read())
    pd_data = pd.DataFrame(data)

    return pd.DataFrame(pd_data['results'].tolist())

url = f'https://educationdata.urban.org/api/v1/college-university/ipeds/directory/{YEAR}/'
directory_df = data_to_pandas(url)
directory_features = [
    "unitid",
    "year",
    "offering_highest_level",
    "inst_size",
    "hbcu",
    "medical_degree",
    "tribal_college",
    "land_grant",
]
directory_data = directory_df[directory_features]


url = f'https://educationdata.urban.org/api/v1/college-university/ipeds/institutional-characteristics/{YEAR}/'
institutional_df = data_to_pandas(url)

institution_features = [
    "unitid",
    "year",
    "inst_affiliation",
    "oncampus_housing",
    "calendar_system",
    "study_abroad",
    "dual_credit",
    "ap_credit",
    "employment_services",
    "placement_services",
    "oncampus_daycare",
    "disability_indicator",
]

institution_data = institutional_df[institution_features]


url = f'https://educationdata.urban.org/api/v1/college-university/ipeds/admissions-enrollment/{YEAR}/'
admissions_stats_df = data_to_pandas(url)
admissions_stats_features = [
    "unitid",
    "year",
    "sex",
    "number_applied",
    "number_admitted",
    "number_enrolled_ft",
    "number_enrolled_pt",
    "number_enrolled_total"
]
admissions_stats_data = admissions_stats_df[admissions_stats_features]


url = f'https://educationdata.urban.org/api/v1/college-university/ipeds/academic-year-tuition/{YEAR}/'
tuition_df = data_to_pandas(url)
url = f'https://educationdata.urban.org/api/v1/college-university/ipeds/academic-year-tuition/{YEAR}/?page=2'
tuition_df2 = data_to_pandas(url)

tuition_features = [
    "unitid",
    "year",
    'level_of_study',
    "tuition_type",
    "tuition_fees_ft",
]
tuition_data = tuition_df[tuition_features]
tuition_data2 = tuition_df2[tuition_features]

# concat
tuition_data = pd.concat([tuition_data, tuition_data2], ignore_index=True)

# groupby 'unitid' and take the mean
tuition_data = tuition_data.groupby('unitid').mean()
tuition_data.reset_index(inplace=True)


url = f'https://educationdata.urban.org/api/v1/college-university/ipeds/academic-year-room-board-other/{YEAR}/'
tuition_board_df = data_to_pandas(url)
tuition_board_features = [
    "unitid",
    "year",
    "living_arrangement",
    "books_supplies",
    "room_board",
    "exp_other"
]

tuition_board_data = tuition_board_df[tuition_board_features]


url = f"https://educationdata.urban.org/api/v1/college-university/ipeds/sfa-grants-and-net-price/{YEAR}/"
financial_aid_df = data_to_pandas(url)
aid_features = [
    "unitid",
    "year",
    'tuition_type',
    'type_of_aid',
    'income_level',
    'average_grant',
    'number_of_students',
    'total_grant',
    'net_price',
    'number_receiving_grants',
]
financial_aid_data = financial_aid_df[aid_features]


url = f"https://educationdata.urban.org/api/v1/college-university/scorecard/earnings/{YEAR}/"
earnings_df = data_to_pandas(url)
features = [
    'unitid',
    'year',
    'years_after_entry',
    'cohort_year',
    'earnings_mean',
    'earnings_sd',
    'earnings_med',
]
earnings_data = earnings_df[features]

# save dataframes to csv
directory_data.to_csv('directory_data.csv', index=False)
institution_data.to_csv('institution_data.csv', index=False)
admissions_stats_data.to_csv('admissions_stats_data.csv', index=False)
tuition_data.to_csv('tuition_data.csv', index=False)
tuition_board_data.to_csv('tuition_board_data.csv', index=False)
financial_aid_data.to_csv('financial_aid_data.csv', index=False)
earnings_data.to_csv('earnings_data.csv', index=False)


#directory_data as the base
merged_data = directory_data

# dataframes to join
dataframes = [
    institution_data,
]

# inner join with each dataframe in the list
for df in dataframes:
    merged_data = pd.merge(merged_data, df, on=['unitid', 'year'], how='inner')

# profile = ProfileReport(merged_data, title="Modeling Features Report")
# profile.to_file('Modeling_feat.html')
# merged_data.to_csv('merged_data.csv', index=False)

# 1. offering_highest_level: Remove rows with -1
data = merged_data[merged_data['offering_highest_level'] != -1]

# 2. inst_size: Remove rows with -2
data = data[data['inst_size'] != -1]
data = data[data['inst_size'] != -2]

# 3. medical_degree: Replace -2 and -1 with 0
data['medical_degree'] = data['medical_degree'].replace([-2, -1], 0)

# 4. inst_affiliation: Remove rows with NaN values
data = data[data['inst_affiliation'].notna()]

# 6. oncampus_housing: Remove rows with NaN values and replace -2 and -1 with 0
data = data[data['oncampus_housing'].notna()]
data['oncampus_housing'] = data['oncampus_housing'].replace([-2, -1], 0)

# 7. calendar_system: Remove rows with NaN values and -2
data = data[data['calendar_system'].notna() & (data['calendar_system'] != -2)]

# Processing remaining columns by replacing NaN, -2 and -1 with 0
columns_to_process = [
    'study_abroad', 'dual_credit', 'ap_credit', 'employment_services',
    'placement_services', 'oncampus_daycare', 'disability_indicator'
]

for col in columns_to_process:
    data = data[data[col].notna()]
    data[col] = data[col].replace([-2, -1], 0)


# profile = ProfileReport(data, title="Modeling Features Report")
# profile.to_file('Modeling_feat_processed.html')
data.to_csv('processed_data_clustering.csv', index=False)
