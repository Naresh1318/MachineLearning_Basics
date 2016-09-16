import sqlite3
import os
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db'
          '(review TEXT, sentiment INTEGER, date TEXT)')
example1 = 'I loved the movie'
c.execute("INSERT INTO review_db"
          "(review, sentiment, date) VALUES"
          "(?, ?, DATE('now'))", (example1, 1))

example2 = 'I did not like the movie'
c.execute("INSERT INTO review_db"
          "(review, sentiment, date) VALUES"
          "(?, ?, DATE('now'))", (example2, 0))

conn.commit()
conn.close()