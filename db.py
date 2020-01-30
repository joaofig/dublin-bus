import sqlite3
import os
import json
import pandas.io.sql as sqlio


class MapDb(object):

    def __init__(self, folder="./db"):
        self.conn_str = os.path.join(folder, 'dublin-bus.sqlite')

    def connect(self):
        return sqlite3.connect(self.conn_str, check_same_thread=False)

    def disconnect(self):
        pass

    def insert_nodes(self, nodes):
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany('''
INSERT INTO node 
    (node_id, version, changeset, timestamp, uid, lat, lon)
values (?,?,?,?,?,?,?)
        ''', nodes)
        conn.commit()
        cur.close()
        conn.close()

    def insert_node_tags(self, node_tags):
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany('''
INSERT INTO  node_tag 
    (node_id, tag_key, tag_value)
values (?,?,?)
        ''', node_tags)
        conn.commit()
        cur.close()
        conn.close()

    def insert_ways(self, ways):
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany('''
INSERT INTO way 
    (way_id, version, changeset, timestamp, uid)
values (?,?,?,?,?)
        ''', ways)
        conn.commit()
        cur.close()
        conn.close()

    def insert_way_nodes(self, way_nodes):
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany('''
INSERT INTO way_node 
    (way_id, node_uid)
values (?,?)
        ''', way_nodes)
        conn.commit()
        cur.close()
        conn.close()

    def insert_way_tagss(self, way_tags):
        conn = self.connect()
        cur = conn.cursor()

        cur.executemany('''
INSERT INTO way_tag
    (way_id, tag_key, tag_value)
values (?,?,?)
        ''', way_tags)
        conn.commit()
        cur.close()
        conn.close()
