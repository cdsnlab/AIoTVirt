#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymongo import MongoClient

from util.Logger import Logger


class DBManager(object):
    def __init__(self, mongo, collection):
        self.logger = Logger()
        self.logger.debug("INTO DBManager!")
        client = MongoClient("mongodb://"+mongo["username"]+":"+mongo["password"]+"@"+mongo["ip"]+":"+mongo["port"])
        database = client.get_database(mongo["database"])
        self.collection = database.get_collection(collection)

    def getCollection(self):
        return self.collection
