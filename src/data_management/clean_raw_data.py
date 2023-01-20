"""
Clean the raw data to use it for the analysis and bring it in the second normal
form: Create a table for countries with IDs, one for time periods with IDs and
one for every data series we do have and use later on. Furthermore, clean and
replace missings for predictors and calculate averages for predictors over the 
years 2012-2014 and safe them in a seperate datafile.

All tested using Jupyter notebook and regression tests.
"""

import pandas as pd
from bld.project_paths import project_paths_join as ppj


def read_in_data(filename):
    """Read in source data."""

    return pd.read_csv(
            ppj('IN_DATA', filename), sep=',')


def create_country_table(data):
    """Create table of countries sorted alphabetically with new IDs."""

    countries = data[['Country']]
    countries = countries.drop_duplicates()
    countries = countries.sort_values(['Country']).reset_index(drop=True)
    countries['country_id'] = countries.index
    countries = countries.set_index('country_id')
    return countries


def create_period_table(data):
    """Create table of Periods with IDs."""

    periods = data[['Period']]
    periods = periods.drop_duplicates()
    periods = periods.reset_index(drop=True)
    periods['period_id'] = periods.index
    periods = periods.set_index('period_id')
    return periods


def create_gdppc_const_prices_table(data, countries, periods):
    """Create table of GDPPC in constant prices (OECD reference year)
    with fixed PPP for every country only including
    country ID and period ID."""

    # Extract necessary columns.
    gdppc_const_prices = data[data['MEASURE'] == 'HVPVOBARSA']
    gdppc_const_prices = gdppc_const_prices[['Country', 'Value', 'Period']]

    # Add country IDs and delete columns we do not need.
    gdppc_const_prices = pd.merge(countries.reset_index(),
                                  gdppc_const_prices.reset_index(),
                                  on=['Country'], how='inner').drop(
                                          ['index', 'Country'], axis=1)

    # Add period IDs, delete columns we do not need and reset Index to IDs.
    gdppc_const_prices = pd.merge(periods.reset_index(),
                                  gdppc_const_prices.reset_index(),
                                  on=['Period'], how='inner').sort_values(
                                          ['index']).set_index(
                                                  ['country_id', 'period_id'])
    gdppc_const_prices = gdppc_const_prices.drop(['index', 'Period'], axis=1)
    return gdppc_const_prices


def create_gdppc_current_prices_table(data, countries, periods):
    """Create table of GDPPC in current prices with current PPP for every country
    only including country ID and period ID."""

    # Extract necessary columns.
    gdppc_current_prices = data[data['MEASURE'] == 'HCPCARSA']
    gdppc_current_prices = gdppc_current_prices[['Country', 'Value', 'Period']]

    # Add country IDs and delete columns we do not need.
    gdppc_current_prices = pd.merge(countries.reset_index(),
                                    gdppc_current_prices.reset_index(),
                                    on=['Country'], how='inner').drop(
                                          ['index', 'Country'], axis=1)

    # Add period IDs, delete columns we do not need and reset Index to IDs.
    gdppc_current_prices = pd.merge(periods.reset_index(),
                                    gdppc_current_prices.reset_index(),
                                    on=['Period'], how='inner').sort_values(
                                          ['index']).set_index(
                                                  ['country_id', 'period_id'])
    gdppc_current_prices = gdppc_current_prices.drop(['index', 'Period'],
                                                     axis=1)
    return gdppc_current_prices


def create_ca_percentage_gdp_table(data, countries, periods):
    """Create table of current accout in percentage of GDP for every country
    only including country ID and period ID."""

    # Extract necessary columns.
    ca_percentage_gdp = data[['Country', 'Time', 'Value']]

    # Rename column for merge.
    ca_percentage_gdp = ca_percentage_gdp.rename(columns={'Time': 'Period'})

    # Merge with countries and periods, droping unnecessary columns.
    ca_percentage_gdp = pd.merge(countries.reset_index(),
                                 ca_percentage_gdp.reset_index(),
                                 on=['Country'], how='inner'
                                 ).drop(['index', 'Country'], axis=1)
    ca_percentage_gdp = pd.merge(periods.reset_index(),
                                 ca_percentage_gdp.reset_index(),
                                 on=['Period'], how='inner').sort_values(
                                 ['index']).set_index(
                                 ['country_id', 'period_id']).drop(
                                 ['index', 'Period'], axis=1)
    return ca_percentage_gdp


def create_table_of_predictors(country_table, predictor_variables):
    """ Create the table of predictor variables for every country only including
    country id and the different predictors."""

    predictors = pd.DataFrame(country_table)

    # Append data for trade openness.
    trade_openness = pd.DataFrame(
            predictor_variables[predictor_variables['Series Name'] ==
                                'Trade (% of GDP)'])
    trade_openness['average'] = (
        trade_openness['2012 [YR2012]'].astype(float) +
        trade_openness['2013 [YR2013]'].astype(float) +
        trade_openness['2014 [YR2014]'].astype(float))/3
    trade_openness = pd.DataFrame(trade_openness[['\ufeffCountry Name',
                                                 'average']])
    trade_openness.rename(
            columns={'\ufeffCountry Name': 'Country',
                     'average': 'trade_openness'},
            inplace=True)
    predictors = pd.merge(predictors.reset_index(),
                          trade_openness.reset_index(),
                          on=['Country'], how='inner').set_index('country_id')
    predictors.drop(['index'], axis=1, inplace=True)

    # Append data for industry share.
    industry_share = pd.DataFrame(
            predictor_variables[predictor_variables['Series Name'] ==
                                'Industry, value added (% of GDP)'])
    # Create missing value for canada 2014 as Canada 2013.
    industry_share['2014 [YR2014]'].loc[19] = \
        industry_share['2013 [YR2013]'].loc[19]
    industry_share['average'] = (
        industry_share['2012 [YR2012]'].astype(float) +
        industry_share['2013 [YR2013]'].astype(float) +
        industry_share['2014 [YR2014]'].astype(float))/3
    industry_share = pd.DataFrame(
        industry_share[['\ufeffCountry Name', 'average']])
    industry_share.rename(
            columns={'\ufeffCountry Name': 'Country',
                     'average': 'industry_share'},
            inplace=True)
    predictors = pd.merge(predictors.reset_index(),
                          industry_share.reset_index(),
                          on=['Country'], how='inner').set_index(
                          'country_id').drop(['index'], axis=1)

    # Append data for inflation rate.
    inflation_rate = pd.DataFrame(
            predictor_variables[predictor_variables['Series Name'] ==
                                'Inflation, consumer prices (annual %)'])
    inflation_rate['average'] = (inflation_rate['2012 [YR2012]'].astype(float)
                                 + inflation_rate['2013 [YR2013]'].astype(float)
                                 + inflation_rate['2014 [YR2014]'].
                                 astype(float))/3
    inflation_rate = pd.DataFrame(inflation_rate[['\ufeffCountry Name',
                                  'average']])
    inflation_rate.rename(
            columns={'\ufeffCountry Name': 'Country', 'average':
                     'inflation_rate'},
            inplace=True)
    predictors = pd.merge(predictors.reset_index(),
                          inflation_rate.reset_index(),
                          on=['Country'], how='inner').set_index(
                          'country_id').drop(['index'], axis=1)

    # Append data for schooling.
    schooling = pd.DataFrame(
            predictor_variables[predictor_variables['Series Name'] ==
                                'Educational attainment, at least completed upper secondary, population 25+, total (%) (cumulative)'])

    # Create missing from older or newer values for: Canada, Chile, Greece,
    # Ireland, Iceland, Japan, Italy, Korea, New Zealand, Norway, Spain,
    # Switzerland
    schooling['2012 [YR2012]'].loc[22] = schooling['2011 [YR2011]'].loc[22]
    schooling['2013 [YR2013]'].loc[22] = schooling['2011 [YR2011]'].loc[22]
    schooling['2014 [YR2014]'].loc[22] = schooling['2011 [YR2011]'].loc[22]
    schooling['2012 [YR2012]'].loc[28] = schooling['2013 [YR2013]'].loc[28]
    schooling['2014 [YR2014]'].loc[28] = schooling['2013 [YR2013]'].loc[28]
    schooling['2012 [YR2012]'].loc[70] = schooling['2014 [YR2014]'].loc[70]
    schooling['2013 [YR2013]'].loc[70] = schooling['2014 [YR2014]'].loc[70]
    schooling['2012 [YR2012]'].loc[82] = schooling['2011 [YR2011]'].loc[82]
    schooling['2013 [YR2013]'].loc[82] = schooling['2011 [YR2011]'].loc[82]
    schooling['2014 [YR2014]'].loc[82] = schooling['2011 [YR2011]'].loc[82]
    schooling['2012 [YR2012]'].loc[88] = 60.7  # Iceland, value from 2005
    schooling['2013 [YR2013]'].loc[88] = 60.7
    schooling['2014 [YR2014]'].loc[88] = 60.7
    schooling['2012 [YR2012]'].loc[94] = schooling['2010 [YR2010]'].loc[94]
    schooling['2013 [YR2013]'].loc[94] = schooling['2010 [YR2010]'].loc[94]
    schooling['2014 [YR2014]'].loc[94] = schooling['2010 [YR2010]'].loc[94]
    schooling['2013 [YR2013]'].loc[100] = schooling['2014 [YR2014]'].loc[100]
    schooling['2012 [YR2012]'].loc[112] = schooling['2010 [YR2010]'].loc[112]
    schooling['2013 [YR2013]'].loc[112] = schooling['2010 [YR2010]'].loc[112]
    schooling['2014 [YR2014]'].loc[112] = schooling['2010 [YR2010]'].loc[112]
    schooling['2013 [YR2013]'].loc[142] = schooling['2014 [YR2014]'].loc[142]
    schooling['2012 [YR2012]'].loc[148] = schooling['2014 [YR2014]'].loc[148]
    schooling['2013 [YR2013]'].loc[148] = schooling['2014 [YR2014]'].loc[148]
    schooling['2013 [YR2013]'].loc[178] = schooling['2014 [YR2014]'].loc[178]
    schooling['2013 [YR2013]'].loc[190] = schooling['2014 [YR2014]'].loc[190]

    schooling['average'] = (schooling['2012 [YR2012]'].astype(float) +
                            schooling['2013 [YR2013]'].astype(float) +
                            schooling['2014 [YR2014]'].astype(float))/3
    schooling = pd.DataFrame(schooling[['\ufeffCountry Name', 'average']])
    schooling.rename(
            columns={'\ufeffCountry Name': 'Country', 'average': 'schooling'},
            inplace=True)
    predictors = pd.merge(predictors.reset_index(), schooling.reset_index(),
                          on=['Country'], how='inner').set_index(
                          'country_id').drop(['index'], axis=1)

    return predictors


if __name__ == "__main__":
    # Read in and create data tables for countries with ID and Periods with ID.
    gdppc_data = read_in_data('GDPPC_OECD_1990_2017.csv')
    countries = create_country_table(gdppc_data)
    periods = create_period_table(gdppc_data)

    # Create data tables for GDPPC including only values, country and period
    # IDs.
    gdppc_const_price = create_gdppc_const_prices_table(
            gdppc_data, countries, periods)
    gdppc_current_price = create_gdppc_current_prices_table(
            gdppc_data, countries, periods)

    # Read in data for CA and create data tables for CA only including values,
    # country ID and period ID
    ca_data = read_in_data(
            'Current_account_percentage_GDP_OECD_quarterly_1990_2017.csv')
    ca_percent_gdp = create_ca_percentage_gdp_table(
            ca_data, countries, periods)

    # Read in data for predictors.
    predictor_table = read_in_data('predictor_data_WDI.csv')
    predictors = create_table_of_predictors(country_table=countries,
                                            predictor_variables=predictor_table
                                            )

    # Save data.
    countries.to_csv(ppj('OUT_DATA', 'countries.csv'))
    periods.to_csv(ppj('OUT_DATA', 'periods.csv'))
    gdppc_const_price.to_csv(ppj('OUT_DATA', 'gdppc_const_prices.csv'))
    gdppc_current_price.to_csv(ppj('OUT_DATA', 'gdppc_current_prices.csv'))
    ca_percent_gdp.to_csv(ppj('OUT_DATA', 'ca_percentage_gdp.csv'))
    predictors.to_csv(ppj('OUT_DATA', 'predictor_data.csv'))
