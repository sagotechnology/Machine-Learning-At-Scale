#!/usr/bin/env python

'''
kafka-topics --zookeeper localhost:2181 --delete --topic impressions
kafka-console-consumer --zookeeper localhost:2181 --topic impressions --from-beginning
'''

import threading, logging, time
import multiprocessing
import json
import collections
import random
from kafka import KafkaConsumer, KafkaProducer

config = json.load(open('click_through_rate_config.json', 'r'))

producer = KafkaProducer(bootstrap_servers='localhost:9092')

counter = collections.Counter()
counter['impression_id'] = 0

def send(impression_id, site, exchange):
	producer.send('impressions', b"%d,%s,%s" % (impression_id, str(site).encode('ASCII'), str(exchange).encode('ASCII')))

time.sleep(3)

for i in range(90):
	for site_exchange, ctr in config['site_exchange_ctr'].items():
		site, exchange = site_exchange.split("+")
		send(counter['impression_id'], site, exchange)
		counter['impression_id'] += 1
	time.sleep(.3)

producer.close()
