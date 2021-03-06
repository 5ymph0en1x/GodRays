import math
import threading
import traceback
from time import sleep
import time
import json
import logging
import urllib
import websocket
from operator import itemgetter
from .auth import expiration_time, generate_signature

# Naive implementation of connecting to BitMEX websocket for streaming real time data.
# The traders still interacts with this as if it were a REST Endpoint, but now it can get
# much more real time data without polling the hell out of the API.
#
# The WebSocket offers a bunch of data as raw properties right on the object.
# On connect, it synchronously asks for a push of all this data then returns.
# Right after, the MM can start using its data. It will be updated in real time, so the MM can
# poll really often if it wants.
class BitMEXWebSocketClient:

	# Don't grow a table larger than this amount. Helps cap memory usage.
	MAX_TABLE_LEN = 1000

	def __init__(self, endpoint, symbols, api_key=None, api_secret=None, subscriptions=None, expiration_seconds=3600):
		'''Connect to the websocket and initialize data stores.'''
		self.logger = logging.getLogger(__name__)
		self.logger.debug("Initializing WebSocket.")

		self.endpoint = endpoint
		self.symbols = symbols

		self.expiration_seconds = expiration_seconds

		if api_key is not None and api_secret is None:
			raise ValueError('api_secret is required if api_key is provided')
		if api_key is None and api_secret is not None:
			raise ValueError('api_key is required if api_secret is provided')

		self.api_key = api_key
		self.api_secret = api_secret

		if subscriptions is not None:
			self.subscription_list = subscriptions
		else:
			self.subscription_list =\
				["execution", "instrument", "margin", "order", "orderBookL2_25", "position", "quote", "trade"]

		self.updates = {}
		self.data = {}
		self.last_data = {}
		self.keys = {}
		self.exited = False

		# We can subscribe right in the connection querystring, so let's build that.
		# Subscribe to all pertinent endpoints
		ws_uri = self.__get_url()
		self.logger.info("Connecting to %s" % ws_uri)
		self.__connect(ws_uri)
		self.logger.info('Connected to WS.')

		# Connected. Wait for partials

		if self.__wait_for_data_arrival(symbols):
			self.logger.info('Got all market data. Starting.')


	@staticmethod
	def _now():
		from datetime import datetime, timezone
		return datetime.now().astimezone(timezone.utc)

	def exit(self):
		'''Call this to exit - will close websocket.'''
		self.ws.close()

	def get_ticker(self, symbol):
		'''Return a ticker object. Generated from quote and trade.'''
		last_quote = self.last_data['quote'][symbol]
		last_trade = self.last_data['trade'][symbol]
		mid = (float(last_quote['bidPrice'] or 0) + float(last_quote['askPrice'] or 0)) / 2
		ticker = {
			"last": last_trade['price'],
			"buy": last_quote['bidPrice'],
			"sell": last_quote['askPrice'],
			"mid": mid
		}
		# The instrument has a tickSize. Use it to round values.
		instrument = self.last_data['instrument'][symbol]
		if 'tickSize' in instrument:
			return {k: round(float(v or 0), -int(math.log10(instrument['tickSize']))) for k, v in ticker.items()}
		else:
			return ticker

	def funds(self, currency):
		'''Get your margin details.'''
		if currency == "XBt":
			return self.data['margin'][0]['withdrawableMargin'], self.data['margin'][0]['walletBalance']
		elif currency == "USDt":
			return self.data['margin'][1]['withdrawableMargin'] / 10**6, self.data['margin'][1]['walletBalance'] / 10**6

	def positions(self):
		'''Get your positions.'''
		return self.data['position']

	def executions(self):
		return self.data['execution']

	def get_order_book_table_name(self):
		if 'orderBookL2' in self.data:
			return 'orderBookL2'
		elif 'orderBookL2_25' in self.data:
			return 'orderBookL2_25'
		else:
			return 'orderBook10'

	def market_depth(self):
		'''Get market depth (orderbook). Returns all levels.'''
		return self.data['orderBookL2_25']

	def orders(self):
		'''Get all your open orders.'''
		orders = self.data['order']
		return orders

	def order_status(self):
		if len(self.data['order']) != 0:
			status = self.data['order']
			proxy = sorted(status, key=itemgetter('timestamp'))
			result = proxy[-1]['ordStatus']
			return result
		else:
			return None

	def open_positions(self):
		'''Get recent trades.'''
		# print("positions", self.data['position'][0]['currentQty'])
		if not self.data['position']:  # For new accounts... Thanks Sam !
			return 0
		else:
			return self.data['position'][0]['currentQty']

	def open_stops(self):
		'''Get recent trades.'''
		# print("stops", self.data['order'])
		return self.data['order']

	def open_orders(self, clOrdIDPrefix):
		'''Get all your open orders.'''
		orders = self.data['order']
		# Filter to only open orders and those that we actually placed
		return [o for o in orders if str(o['clOrdID']).startswith(clOrdIDPrefix) and order_leaves_quantity(o)]

	def recent_trades(self):
		'''Get recent trades.'''
		return self.data['trade']

	#
	# End Public Methods
	#

	def __connect(self, wsURL):
		'''Connect to the websocket in a thread.'''
		self.logger.debug("Starting thread")

		self.ws = websocket.WebSocketApp(wsURL,
												on_message=self.__on_message,
												on_close=self.__on_close,
												on_open=self.__on_open,
												on_error=self.__on_error,
												header=self.__get_auth())

		self.wst = threading.Thread(target=lambda: self.ws.run_forever())
		self.wst.daemon = True
		self.wst.start()
		self.logger.debug("Started thread")

		# Wait for connect before continuing
		conn_timeout = 5
		while not self.ws.sock or not self.ws.sock.connected and conn_timeout:
			sleep(1)
			conn_timeout -= 1
		if not conn_timeout:
			self.logger.error("Couldn't connect to WS! Exiting.")
			self.exit()
			raise websocket.WebSocketTimeoutException('Couldn\'t connect to WS! Exiting.')

	def __get_auth(self):
		'''Return auth headers. Will use API Keys if present in settings.'''
		if self.api_key:
			self.logger.info("Authenticating with API Key.")
			# To auth to the WS using an API key, we generate a signature of a nonce and
			# the WS API endpoint.
			expires = expiration_time(self.expiration_seconds)
			return [
				"api-expires: " + str(expires),
				"api-signature: " + generate_signature(self.api_secret, 'GET', '/realtime', expires, ''),
				"api-key:" + self.api_key
			]
		else:
			self.logger.info("Not authenticating.")
			return []

	def __get_url(self):
		import copy

		'''
		Generate a connection URL. We can define subscriptions right in the querystring.
		Most subscription topics are scoped by the symbol we're listening to.
		'''

		# You can sub to orderBookL2 for all levels, or orderBook10 for top 10 levels & save bandwidth
		subscriptions_per_symbol = copy.copy(self.subscription_list)
		if "margin" in subscriptions_per_symbol:
			subscriptions_per_symbol.remove("margin")
			generic_subscriptions = ["margin"]
		else:
			generic_subscriptions = []

		subscriptions = [
			','.join(
				[sub + ':' + symbol for symbol in self.symbols]
			) 
			for sub in subscriptions_per_symbol
		]
		subscriptions += generic_subscriptions

		uri_parts = list(urllib.parse.urlparse(self.endpoint))
		uri_parts[0] = uri_parts[0].replace('http', 'ws')
		uri_parts[2] = "/realtime?subscribe={}".format(','.join(subscriptions))

		return urllib.parse.urlunparse(uri_parts)

	def __wait_for_data_arrival(self, symbols):
		'''On subscribe, this data will come down. Wait for it.'''
		targets = set(self.subscription_list)
		t = time.time()
		while not targets <= set(self.data):
			sleep(0.1)
			if time.time() - t > 60:
				self.logger.error("Error: data is not coming")
				self.exit()
				return False
		return True

	def __send_command(self, command, args=None):
		'''Send a raw command.'''
		if args is None:
			args = []
		self.ws.send(json.dumps({"op": command, "args": args}))

	def __on_message(self, dummy1, message):
		'''Handler for parsing WS messages.'''
		message = json.loads(message)
		self.logger.debug(json.dumps(message))

		table = message.get('table')
		action = message.get('action')
		# Remember the time of update.
		if table is not None and 0 < len(table):
			self.updates[table] = self._now()
		try:
			if 'subscribe' in message:
				self.logger.debug("Subscribed to %s." % message['subscribe'])
			elif action:
				if table not in self.data:
					self.data[table] = []
					self.last_data[table] = {}

				# There are four possible actions from the WS:
				# 'partial' - full table image
				# 'insert'  - new row
				# 'update'  - update row
				# 'delete'  - delete row
				if action == 'partial':
					self.logger.debug("%s: partial" % table)
					self.data[table] += message['data']
					if table in ['quote', 'trade', 'instrument']:
						for row in message['data']:
							self.last_data[table][row['symbol']] = row
					# Keys are communicated on partials to let you know how to uniquely identify
					# an item. We use it for updates.
					self.keys[table] = message['keys']
				elif action == 'insert':
					self.logger.debug('%s: inserting %s' % (table, message['data']))
					self.data[table] += message['data']

					if table in ['quote', 'trade', 'instrument']:
						for row in message['data']:
							self.last_data[table][row['symbol']] = row
					# Limit the max length of the table to avoid excessive memory usage.
					# Don't trim orders because we'll lose valuable state if we do.
					if table not in ['order', 'orderBookL2'] and len(self.data[table]) > BitMEXWebSocketClient.MAX_TABLE_LEN:
						self.data[table] = self.data[table][BitMEXWebSocketClient.MAX_TABLE_LEN // 2:]
				elif action == 'update':
					self.logger.debug('%s: updating %s' % (table, message['data']))
					# Locate the item in the collection and update it.
					for updateData in message['data']:
						item = find_by_keys(self.keys[table], self.data[table], updateData)
						if not item:
							return  # No item found to update. Could happen before push
						item.update(updateData)
						# Remove cancelled / filled orders
						if table == 'order' and not order_leaves_quantity(item):
							self.data[table].remove(item)
				elif action == 'delete':
					self.logger.debug('%s: deleting %s' % (table, message['data']))
					# Locate the item in the collection and remove it.
					for deleteData in message['data']:
						item = find_by_keys(self.keys[table], self.data[table], deleteData)
						self.data[table].remove(item)
				else:
					raise Exception("Unknown action: %s" % action)
		except:
			self.logger.error(traceback.format_exc())

	def __on_error(self, dummy2, error):
		'''Called on fatal websocket errors. We restart on these.'''
		self.logger.error("Error : %s" % error)
		self.ws.close()
		self.exited = True
		time.sleep(1)
		self.__connect(self.__get_url())

	def __on_open(self, dummy3):
		'''Called when the WS opens.'''
		self.logger.debug("WebSocket Opened.")

	def __on_close(self, dummy4a, dummy4b, dummy4c):
		'''Called on websocket close.'''
		self.logger.info('WebSocket Closed')


# Utility method for finding an item in the store.
# When an update comes through on the websocket, we need to figure out which item in the array it is
# in order to match that item.
#
# Helpfully, on a data push (or on an HTTP hit to /api/v1/schema), we have a "keys" array. These are the
# fields we can use to uniquely identify an item. Sometimes there is more than one, so we iterate through all
# provided keys.
def find_by_keys(keys, table, matchData):
	for item in table:
		if all(item[k] == matchData[k] for k in keys):
			return item


def order_leaves_quantity(o):
	if o['leavesQty'] is None:
		return True
	return o['leavesQty'] > 0
