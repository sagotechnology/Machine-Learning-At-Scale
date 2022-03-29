#!/usr/bin/env python

'''
kafka-topics --zookeeper localhost:2181 --delete --topic clicks
kafka-console-consumer --zookeeper localhost:2181 --topic clicks --from-beginning
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

def send(impression_id):
	producer.send('clicks', b"%d" % (impression_id))

time.sleep(3)

for i in range(90):
	for site_exchange, ctr in config['site_exchange_ctr'].items():
		if random.random() <= ctr: 
			send(counter['impression_id'])
		counter['impression_id'] += 1
	time.sleep(.3)

producer.close()
