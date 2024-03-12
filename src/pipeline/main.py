import pyarrow.parquet as pq
from prefect import flow, get_run_logger, task
import pandas as pd
from typing import Dict, Tuple
from io import StringIO
import csv
from sqlalchemy import create_engine
import asyncio

def generate_log(data: any):
    logger = get_run_logger()
    logger.info(data)


@task
def retrieve_from_local(path: str = "../../data/yellow_tripdata_2014-01.parquet") -> pd.DataFrame:
    generate_log("Retrieving")
    trips = pq.read_table(path)
    df = trips.to_pandas()
    return df


@task
def clean_trips_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df['trip_id'] = df.index
    df.fillna({'airport_fee': 0}, inplace=True)
    df.dropna(subset=['store_and_fwd_flag'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
    generate_log(df.head(10))


@flow
def extract(path: str = "../../data/yellow_tripdata_2014-01.parquet") -> pd.DataFrame:
    df = retrieve_from_local(path)
    df = clean_trips_data(df)
    return df


@task
def datetime_dimension(df: pd.DataFrame) -> pd.DataFrame:
    generate_log("Transforming datetime dimension")
    datetime_dim = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].reset_index(drop=True)
    datetime_dim['pick_hour'] = datetime_dim['tpep_pickup_datetime'].dt.hour
    datetime_dim['pick_day'] = datetime_dim["tpep_pickup_datetime"].dt.day
    datetime_dim['pick_month'] = datetime_dim['tpep_pickup_datetime'].dt.month
    datetime_dim['pick_year'] = datetime_dim["tpep_pickup_datetime"].dt.year
    datetime_dim['pick_weekday'] = datetime_dim["tpep_pickup_datetime"].dt.weekday

    datetime_dim['drop_hour'] = datetime_dim['tpep_dropoff_datetime'].dt.hour
    datetime_dim['drop_day'] = datetime_dim["tpep_dropoff_datetime"].dt.day
    datetime_dim['drop_month'] = datetime_dim['tpep_dropoff_datetime'].dt.month
    datetime_dim['drop_year'] = datetime_dim["tpep_dropoff_datetime"].dt.year
    datetime_dim['drop_weekday'] = datetime_dim["tpep_dropoff_datetime"].dt.weekday

    datetime_dim['datetime_id'] = datetime_dim.index
    datetime_dim = datetime_dim[
        ['datetime_id',
         'tpep_pickup_datetime', 'pick_hour', 'pick_day', 'pick_month', 'pick_year', 'pick_weekday',
         'tpep_dropoff_datetime', 'drop_hour', 'drop_day', 'drop_month', 'drop_year', 'drop_weekday']]
    generate_log(datetime_dim.head(10))
    return datetime_dim


@task
def passenger_count_dimension(df: pd.DataFrame) -> pd.DataFrame:
    generate_log("Transforming passenger count dimension")
    passenger_count_dim = df[['passenger_count']].reset_index(drop=True)
    passenger_count_dim['passenger_count_id'] = passenger_count_dim.index
    passenger_count_dim = passenger_count_dim[['passenger_count_id', 'passenger_count']]
    generate_log(passenger_count_dim.head(10))
    return passenger_count_dim


@task
def trip_distance_dimension(df: pd.DataFrame) -> pd.DataFrame:
    generate_log("Transforming trip distance dimension")
    trip_distance_dim = df[['trip_distance']].reset_index(drop=True)
    trip_distance_dim['trip_distance_id'] = trip_distance_dim.index
    trip_distance_dim = trip_distance_dim[['trip_distance_id', 'trip_distance']]
    generate_log(trip_distance_dim.head(10))
    return trip_distance_dim


@task
def rate_code_dimension(df: pd.DataFrame) -> pd.DataFrame:
    generate_log("Transforming rate code dimension")
    rate_code_dim = df[['RatecodeID']].reset_index(drop=True)
    rate_code_dim['rate_code_id'] = rate_code_dim.index
    rate_code_name = {
        1: "Standard Rate",
        2: "JFK",
        3: "Newark",
        4: "Nassau or Westchester",
        5: "Negotiated fare",
        6: "Group ride"
    }

    rate_code_dim['rate_code_name'] = rate_code_dim['RatecodeID'].map(rate_code_name)
    rate_code_dim = rate_code_dim[['rate_code_id', 'RatecodeID', 'rate_code_name']]
    generate_log(rate_code_dim.head(10))
    return rate_code_dim


@task
def payment_type_dimension(df: pd.DataFrame) -> pd.DataFrame:
    generate_log("Transforming payment type dimension")
    payment_type_dim = df[['payment_type']].reset_index(drop=True)
    payment_type_dim['payment_type_id'] = payment_type_dim.index
    payment_type_name = {
        1: "Credit card",
        2: "Cash",
        3: "No Charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided trip"
    }
    payment_type_dim['payment_type_name'] = payment_type_dim['payment_type'].map(payment_type_name)
    payment_type_dim = payment_type_dim[['payment_type_id', 'payment_type_name', 'payment_type']]
    generate_log(payment_type_dim.head(10))
    return payment_type_dim


@flow
def transform() -> Tuple[pd.DataFrame, Dict]:
    df = extract()
    datetime_data = datetime_dimension(df)
    passenger_count_data = passenger_count_dimension(df)
    trip_distance_data = trip_distance_dimension(df)
    rate_code_data = rate_code_dimension(df)
    payment_type_data = payment_type_dimension(df)
    return df, {"datetime_dimension": datetime_data,
                "passenger_count_dimension": passenger_count_data,
                "trip_distance_dimension": trip_distance_data,
                "rate_code_dimension": rate_code_data,
                "payment_type_dimension": payment_type_data}


@flow
def merge() -> Tuple[Dict, pd.DataFrame]:
    df, data = transform()
    fact_table = (df.
                  merge(data["datetime_dimension"], left_on='trip_id', right_on='datetime_id').
                  merge(data["passenger_count_dimension"], left_on='trip_id', right_on='passenger_count_id').
                  merge(data["trip_distance_dimension"], left_on='trip_id', right_on='trip_distance_id').
                  merge(data["payment_type_dimension"], left_on='trip_id', right_on='payment_type_id').
                  merge(data["rate_code_dimension"], left_on='trip_id', right_on='rate_code_id'))[[
        'trip_id', 'VendorID', 'datetime_id', 'passenger_count_id', 'trip_distance_id',
        'rate_code_id', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
        'payment_type_id', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'improvement_surcharge', 'total_amount'
    ]]
    generate_log("Merging all the dimension tables")
    generate_log(fact_table.head())
    return data, fact_table


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Execute SQL statement inserting data

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join(['"{}"'.format(k) for k in keys])
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


@task
async def load_to_table(conn, df, table_name):
    df.to_sql(table_name,
              conn,
              method=psql_insert_copy,
              index=False,
              if_exists='replace')


@flow
async def load():
    data, df = merge()
    generate_log("Inserting data into datetime_dimension")
    engine = create_engine('postgresql://postgres:adri@localhost:54320/postgres')

    tasks = []

    for table in data:
        tasks.append(load_to_table(engine, data[table], table))

    tasks.append(load_to_table(engine, df, 'fact_table'))

    await asyncio.gather(*tasks)


async def main():
    await load.serve("load-data")


if __name__ == "__main__":
    asyncio.run(main())
