import sqlite3
import pandas as pd
import numpy as np

from tflite_support.task import processor
FeatureVector = processor.FeatureVector

class DataBase:
    def __init__(self, dbname, csv):
        self.df = pd.read_csv(csv)
        self.conn = sqlite3.connect(dbname)
        self.df.to_sql("imagedb", self.conn, if_exists="replace")
        self.cur = self.conn.cursor()

    def get_db(self, offset):
        select_sql = "SELECT * FROM imagedb LIMIT 1 OFFSET " + str(offset)
        #print(select_sql)
        for row in self.cur.execute(select_sql):
            vector = np.array(row[1:1281])
        return FeatureVector(vector)

def test():
    dbname = "../data/images/amalab/lab_root_3/loop_2/color/DATABASE.db"
    #dbname = "../data/logs/DATABASE.db"
    csv = "../data/images/amalab/lab_root_3/loop_2/color/feature_vector.csv"
    #csv = "../data/logs/searcher_result.csv"
    conn = make_db(dbname, csv)
    cur = conn.cursor()
    select_sql = "SELECT * FROM imagedb LIMIT 1 OFFSET 0"
    #select_sql = "PRAGMA table_info(imagedb)"
    #select_sql = "SELECT COUNT(*) FROM imagedb"
    for row in cur.execute(select_sql):
        #print(row)
        vector = np.array(row[1:1281])
        print(vector)
        print(vector[-1])
        #vector.pop(0)
        print(len(vector))
    cur.close()
    conn.close()

def func_test():
    dbname = "../data/images/amalab/lab_root_3/loop_2/color/DATABASE.db"
    #dbname = "../data/logs/DATABASE.db"
    csv = "../data/images/amalab/lab_root_3/loop_2/color/feature_vector.csv"
    #csv = "../data/logs/searcher_result.csv"
    db = DataBase(dbname, csv)
    vector = db.get_db(0)
    print(vector)


def main():
    func_test()

if __name__ == "__main__":
    main()