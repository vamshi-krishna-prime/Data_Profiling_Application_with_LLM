"""
Module for querying SQL database
@author: vamsi krishna
"""

from database_connector import DataPullerBase
import math
from threading import Thread
import psycopg2 as pg
from sqlalchemy import create_engine
import json
import pandas as pd

import logging

class PostgresConnector(DataPullerBase):

    def __init__(self, credentials):
        '''
        Init object

        Params
        ------------
        credentials: dict in format {'username': ****** ,'password':******}, or path to json file with the same format
        '''
        if isinstance(credentials, str):
            with open(credentials) as f:
                self._credentials = json.load(f)
        else:
            self._credentials = credentials

        self._engine = None

    def _init_connection(self):
        pass

    def _create_engine(self):
        if self._engine:
            self._dispose_engine()
        self._engine = create_engine('postgresql+psycopg2://' + self._credentials['username'] + ':' + self._credentials['password'] + self._credentials['uri'])

    def _get_connection(self):
        return self._engine.connect()

    def _dispose_engine(self):
        self._engine.dispose()
        self._engine = None

    def get_engine(self):
        '''
        Initialize an engine to fire a connection to database

        Return
        ------
        sqlalchemy.engine.Engine
        '''
        return create_engine('postgresql+psycopg2://' + self._credentials['username'] + ':' + self._credentials['password'] + self._credentials['uri'])

    def data_pull(self, query):
        '''
        Execute given query

        Parameters
        -------
        query: str - Query to execute

        Returns
        ----------
        pd.DataFrame - Extracted data
        '''

        engine = self.get_engine()

        sql_reader = pd.read_sql(query, con=engine, chunksize=10 ** 4)

        query_result_df = None
        for pulled_data_df in sql_reader:
            if query_result_df is None:
                query_result_df = pulled_data_df
            else:
                query_result_df = pd.concat([query_result_df, pulled_data_df], axis=0)

        engine.dispose()

        return query_result_df

    def _pull_list_simple(self, q, ids_list, res=None, con_sse=None, chunk_size=500):
        '''
        Pull a list of rows given a query

        Parameters
        ----------
        query: str - The query with rows to be inserted
        ids_list: list - List with the selected rows to be extracted
        chunk_size: int - Size of query batches
        res: list - optional list to append the results to
        con_sse: pg con - optional connection to Postgres

        Return
        ----------
        list: Results from batches appended in a list

        '''

        dispose_engine = False
        if con_sse is None:
            if self._engine:
                con_sse = self._get_connection()
            else:
                dispose_engine = True
                self._create_engine()
                con_sse = self._get_connection()
        if res is None: res = []

        chunks = math.ceil(len(ids_list)/chunk_size)

        for i in range(0, chunks):

            try:
                id_sub_list = ids_list[i*chunk_size:(i+1)*chunk_size]
                id_sub_list = "('" + "', '".join([str(_) for _ in id_sub_list]) + "')"
                t_query = q.replace('()', id_sub_list)
                df = pd.read_sql(t_query, con_sse)
                if(len(df)>0): res.append(df)

            except Exception as error:
                logging.error(error)
                pass

        if dispose_engine:
            self._dispose_engine()

        return res

    def _threaded_process(self, id_range, q, func, nthreads=4, con_sse=None, chunk_size=500):
        '''
        Process the query in a specified number of threads

        Parameters
        ----------
        id_range: list - List with the ids to be extracted
        q: str - The query with rows to be inserted
        func: function - Reference to the function that will process the query
        nthreads: int - Number of threads to run the query on
        chunk_size: int - Size of query batches
        con_sse: pg con - optional connection to Postgres

        Return
        ----------
        list - Appended results of whatever the passed function returns
        '''
        res = []
        threads = []

        use_external_con = con_sse is not None
        dispose_engine = False
        if self._engine is None and not use_external_con:
            self._create_engine()


        # create the threads
        for i in range(nthreads):
            ids = id_range[i::nthreads]
            if not use_external_con:
                con_sse = self._get_connection()
            t = Thread(target=func, args=(q, ids, res, con_sse, chunk_size))
            threads.append(t)

        # start the threads
        [ t.start() for t in threads ]
        # wait for the threads to finish
        [ t.join() for t in threads ]

        try:
            con_sse.close()
        except Exception as e:
            logging.error('Could not close connection')
            logging.error(e)
            pass

        if dispose_engine:
            self._dispose_engine()

        return res

    def pull_list(self, query, ids_list, nthreads=4, chunk_size=500):
        '''
        Pull a list of rows given a query

        Parameters
        ----------
        query: str - The query with the specified rows to be inserted
        ids_list: list - List with the ids to be extracted
        nthreads: int - Number of threads to run the query on
        chunk_size: int - Size of query batches

        Return
        ----------
        pd.DataFrame
            Results from batches concatennated in a DataFrame
        '''

        assert nthreads > 0 and type(nthreads)==int , "Please specify a positive integer as the number of threads"
        if nthreads == 1:
            res = self._pull_list_simple(query, ids_list, chunk_size=chunk_size)
        else:
            res = self._threaded_process(ids_list, query, self._pull_list_simple, nthreads=nthreads, chunk_size=chunk_size)

        if len(res)>0:
            res = pd.concat(res).reset_index(drop=True)
        else:
            res = pd.DataFrame()

        return res


    def append_rows_to_db(self, df, table_name, chunk_size=500):
        '''
        Append new rows to the given table

        Parameters
        ---------------
        df: pd.DataFrame - Dataframe with the rows with want to append
        table_name: str - Name of the table in the DB
        chunk_size: int - Size of query batches
        '''

        dispose_engine = False

        if not self._engine:
            dispose_engine = True
            self._create_engine()


        chunks = math.ceil(len(df)/chunk_size)
        ids = list(df.index)

        for i in range(0, chunks):
            logging.debug(f"Processing {i+1} out of {chunks}")
            try:
                id_sub_list = ids[i*chunk_size:(i+1)*chunk_size]
                temp_df = df.reindex(id_sub_list)

                con_sse = self._engine.raw_connection()
                cursor=con_sse.cursor()

                args_str = b','.join(cursor.mogrify(f"({','.join(['%s']*len(temp_df.columns))})", x[1:]) for x in temp_df.itertuples())

                cursor.execute(f"INSERT INTO {table_name} VALUES ".encode("ascii", "ignore") + args_str)
                con_sse.commit()
                cursor.close()

            except Exception as error:
                logging.error(error)
                pass

        if dispose_engine:
            self._dispose_engine()
