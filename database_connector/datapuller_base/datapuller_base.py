"""
Module for pulling data from a database - inheretance
@author: vamsi krishna
"""

import abc

class DataPullerBase(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and
                callable(subclass.load_data_source) and
                hasattr(subclass, 'extract_text') and
                callable(subclass.extract_text) or
                NotImplemented)

    @abc.abstractmethod
    def _init_connection(self, credentials):
        '''
        Create a connection to the database
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def data_pull(self, query):
        '''
        Execute the given query
        '''
        raise NotImplementedError
