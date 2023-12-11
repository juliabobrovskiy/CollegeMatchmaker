from urllib.request import urlopen
from json import loads
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

YEAR = 2014
years = range(2001, 2019)
def data_to_pandas(url):
    response = urlopen(url)
    data = loads(response.read())
    pd_data = pd.DataFrame(data)

    return pd.DataFrame(pd_data['results'].tolist())


def get_directory_data_for_year(year):
    api_url = f"https://educationdata.urban.org/api/v1/college-university/ipeds/directory/{year}/"
    all_data = []

    while api_url:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data['results'])
            api_url = data['next']
        else:
            print(f"Error: Unable to retrieve directory data for {year}")
            break

    return all_data

# Example usage
all_years_directory_data = []
for year in years:
    data = get_directory_data_for_year(year)
    for item in data:
        item['year'] = year
    all_years_directory_data.extend(data)

# Convert to pandas dataframe
directory_stats_df = pd.DataFrame(all_years_directory_data)

directory_features = [
    "unitid",
    "offering_highest_level", "offering_highest_degree", 'offering_grad',
    "inst_size",
    "hbcu",
    "medical_degree",
    "tribal_college",
    "land_grant",
    'sector', 'inst_control',
    'fips'
]
directory_stats_df = directory_stats_df[directory_features]
directory_stats_df = directory_stats_df.groupby('unitid').mean().reset_index()

# fill missing values with mean
directory_data = directory_stats_df.fillna(directory_stats_df.mean())


def get_institutional_data_for_year(year):
    api_url = f"https://educationdata.urban.org/api/v1/college-university/ipeds/institutional-characteristics/{year}/"
    all_data = []

    while api_url:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data['results'])
            api_url = data['next']
        else:
            print(f"Error: Unable to retrieve institutional characteristics data for {year}")
            break

    return all_data

# Example usage
all_years_institutional_data = []
for year in years:
    data = get_institutional_data_for_year(year)
    for item in data:
        item['year'] = year
    all_years_institutional_data.extend(data)

# Convert to pandas dataframe
institutional_stats_df = pd.DataFrame(all_years_institutional_data)

institution_features = [
    "unitid",
    # "year",
    "inst_affiliation",
    "oncampus_housing",
    "calendar_system",
    "study_abroad",
    "dual_credit",
    "ap_credit",
    "employment_services",
    "placement_services",
    "oncampus_daycare",
    "disability_indicator", 'disability_percentage',
    'cont_prof_prog_offered', 'occupational_prog_offered'
]
institutional_stats_df = institutional_stats_df[institution_features]
institutional_stats_df = institutional_stats_df.groupby('unitid').mean().reset_index()

# fill missing values with mean
institutional_data = institutional_stats_df.fillna(institutional_stats_df.mean())

def get_admission_data_for_year(year):
    api_url = f"https://educationdata.urban.org/api/v1/college-university/ipeds/admissions-enrollment/{year}/"
    all_data = []

    while api_url:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data['results'])
            api_url = data['next']  # Gets the URL for the next page of results, if it exists
        else:
            print(f"Error: Unable to retrieve data for {year}")
            break

    return all_data

# Example usage
all_years_data = []
for year in years:
    data = get_admission_data_for_year(year)
    for item in data:
        item['year'] = year  # Add a year field to each item
    all_years_data.extend(data)

# convert to pandas dataframe
admissions_stats_df = pd.DataFrame(all_years_data)

sum_columns = ['number_applied', 'number_admitted', 'number_enrolled_ft',
               'number_enrolled_pt', 'number_enrolled_total']

admissions_stats_df = admissions_stats_df.groupby(['unitid', 'fips']).agg({col: 'sum' for col in sum_columns}).reset_index()
admissions_stats_df['acceptance_rate'] = admissions_stats_df['number_admitted'] / admissions_stats_df['number_applied']

admissions_stats_data = admissions_stats_df[['unitid', 'acceptance_rate']]
admissions_stats_data = admissions_stats_data[['unitid', 'acceptance_rate']].fillna(admissions_stats_data.mean())
admissions_stats_data = admissions_stats_data.replace(0, admissions_stats_data.mean())


def get_tuition_data_for_year(year):
    api_url = f"https://educationdata.urban.org/api/v1/college-university/ipeds/academic-year-tuition/{year}/"
    all_data = []

    while api_url:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data['results'])
            api_url = data['next']
        else:
            print(f"Error: Unable to retrieve tuition data for {year}")
            break

    return all_data

# Example usage
all_years_tuition_data = []
for year in years:
    data = get_tuition_data_for_year(year)
    for item in data:
        item['year'] = year
    all_years_tuition_data.extend(data)

# Convert to pandas dataframe
tuition_stats_df = pd.DataFrame(all_years_tuition_data)

tuition_features = ['unitid', 'tuition_fees_ft']
tuition_stats_df = tuition_stats_df[tuition_features]

# Group by 'unitid' and calculate the mean
tuition_stats_df = tuition_stats_df.groupby('unitid').mean().reset_index()

# fill missing values with mean
tuition_stats_df = tuition_stats_df.fillna(tuition_stats_df.mean())

# fill zero values with mean
tuition_data = tuition_stats_df.replace(0, tuition_stats_df.mean())


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
institutional_data.to_csv('institution_data.csv', index=False)
admissions_stats_data.to_csv('admissions_stats_data.csv', index=False)
tuition_data.to_csv('tuition_data.csv', index=False)
tuition_board_data.to_csv('tuition_board_data.csv', index=False)
financial_aid_data.to_csv('financial_aid_data.csv', index=False)
earnings_data.to_csv('earnings_data.csv', index=False)


#directory_data as the base
merged_data = directory_data

# dataframes to join
dataframes = [
    institutional_data,
    admissions_stats_data,
    tuition_data,
]

# inner join with each dataframe in the list
for df in dataframes:
    merged_data = pd.merge(merged_data, df, on=['unitid'], how='inner')

# profile = ProfileReport(merged_data, title="Modeling Features Report")
# profile.to_file('Modeling_feat.html')
# merged_data.to_csv('merged_data.csv', index=False)

# cleanup

merged_data.loc[merged_data['inst_size'] < 0, 'inst_size'] = 1
merged_data.loc[merged_data['medical_degree'] < 0, 'medical_degree'] = 0
merged_data.loc[merged_data['disability_indicator'] < 0, 'disability_indicator'] = 0
merged_data.loc[merged_data['cont_prof_prog_offered'] < 0, 'cont_prof_prog_offered'] = 0
merged_data.loc[merged_data['occupational_prog_offered'] < 0, 'occupational_prog_offered'] = 0
merged_data.loc[merged_data['offering_highest_level'] < 0, 'offering_highest_level'] = 0
merged_data.loc[merged_data['offering_highest_degree'] < 0, 'offering_highest_degree'] = 0
merged_data.loc[merged_data['offering_grad'] < 0, 'offering_grad'] = 0
merged_data.loc[merged_data['hbcu'] < 0, 'hbcu'] = 0
merged_data.loc[merged_data['tribal_college'] < 0, 'tribal_college'] = 0
merged_data.loc[merged_data['oncampus_housing'] < 0, 'oncampus_housing'] = 0
merged_data.loc[merged_data['oncampus_daycare'] < 0, 'oncampus_daycare'] = 0
merged_data.loc[merged_data['study_abroad'] < 0, 'study_abroad'] = 0
merged_data.loc[merged_data['dual_credit'] < 0, 'dual_credit'] = 0
merged_data.loc[merged_data['ap_credit'] < 0, 'ap_credit'] = 0
merged_data.loc[merged_data['employment_services'] < 0, 'employment_services'] = 0
merged_data.loc[merged_data['placement_services'] < 0, 'placement_services'] = 0

# profile = ProfileReport(data, title="Modeling Features Report")
# profile.to_file('Modeling_feat_processed.html')
merged_data.to_csv('processed_data_clustering.csv', index=False)
