import requests
import hmac
import hashlib
import time

class API():

    def __init__(self, api_key, api_secret, shutdown_callback=None):
        
        self.api_key = api_key
        self.api_secret = api_secret

        self.error_counter = 0

        self.error_level_one_wait_time = 0.01
        self.error_level_two_wait_time = 0.1
        self.error_level_three_wait_time = 1

        # wait times for requests
        self.wait_for_websocket = 1
        self.get_candlestick_wait = 0.01
        self.get_ticker_wait = 0.01
        self.get_user_balance_wait = 0.01 
        self.create_order_wait = 0.007
        self.cancel_order_wait = 0.007
        self.cancel_all_orders_wait = 0.007
        self.create_order_list_wait = 0.007 
        self.get_order_history_wait = 0.01
        self.get_instruments_wait = 0.01

        self.shutdown_callback = shutdown_callback

    def params_to_str(self, obj, level):
        MAX_LEVEL = 3
        if level >= MAX_LEVEL:
            return str(obj)

        return_str = ""
        for key in sorted(obj):
            return_str += key
            if obj[key] is None:
                return_str += 'null'
            elif isinstance(obj[key], list):
                for subObj in obj[key]:
                    return_str += self.params_to_str(subObj, ++level)
            else:
                return_str += str(obj[key])

        return return_str

    def error_handler(self):

        self.error_counter += 1
        if (self.error_counter < 3):
            time.sleep(self.error_level_one_wait_time)
        elif (self.error_counter < 6):
            time.sleep(self.error_level_two_wait_time)
        elif (self.error_counter < 9):
            time.sleep(self.error_level_three_wait_time)   
        elif (self.error_counter == 9 and self.shutdown_callback):
            self.shutdown_callback()
            

    def get_candlestick(self, name, timeframe, start_ts, end_ts): # NOTE: Only 300 candles can be requested at a time
        url = f'https://api.crypto.com/exchange/v1/public/get-candlestick?instrument_name={name}&timeframe={timeframe}&count=300&start_ts={start_ts}&end_ts={end_ts}'

        while self.error_counter < 9:
            try:
                response = requests.get(url)
                time.sleep(self.get_candlestick_wait)
                if response.status_code == 200:
                    self.error_counter = 0
                    return response.json()
            
                self.error_handler()

            except requests.exceptions.RequestException as e:
                self.error_handler()

    def get_ticker(self, name):
        url = f'https://api.crypto.com/v2/public/get-tickers?instrument_name={name}'

        while self.error_counter < 9:
            try:
                response = requests.get(url)
                time.sleep(self.get_ticker_wait)
                if response.status_code == 200:
                    self.error_counter = 0
                    return response.json()
                
                self.error_handler()

            except requests.exceptions.RequestException as e:
                self.error_handler()

    def get_user_balance(self):

        req = {
            "id": 11,
            "nonce" : int(time.time() * 1000),
            "method": "private/user-balance",
            "api_key": self.api_key,
            "params": {}
        }

        paramString = ""

        if "params" in req:
            for key in req['params']:
                paramString += key
                paramString += str(req['params'][key])

        sigPayload = req['method'] + str(req['id']) + req['api_key'] + paramString + str(req['nonce'])
        
        req['sig'] = hmac.new(
            bytes(str(self.api_secret), 'utf-8'),
            msg=bytes(sigPayload, 'utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

        while self.error_counter < 9:
            try:
                response = requests.post('https://api.crypto.com/exchange/v1/private/user-balance', json=req) # THIS IS THE CORRECT URL FOR OLDER VERSIONS OF THE API
                time.sleep(self.get_user_balance_wait)
                if response.status_code == 200:
                    self.error_counter = 0
                    return response.json()
                self.error_handler()
            except requests.exceptions.RequestException as e:
                self.error_handler()

    
    def create_order(self, name, order_type, price, quantity): # individual orders
        req = {
            {
            "id": 1,
            "nonce" : int(time.time() * 1000),
            "method": "private/create-order",
            "params": {
                "instrument_name": name,
                "side": order_type,
                "type": "MARKET",
                "price": price,
                "quantity": quantity
                }
            }
        }

        paramString = ""

        if "params" in req:
            for key in req['params']:
                paramString += key
                paramString += str(req['params'][key])
    
        sigPayload = req['method'] + str(req['id']) + paramString + str(req['nonce'])

        req['sig'] = hmac.new(
            bytes(str(self.api_secret), 'utf-8'),
            msg=bytes(sigPayload, 'utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

        while self.error_counter < 9:
            try:
                response = requests.post('https://api.crypto.com/exchange/v1/private/create-order', json=req)
                time.sleep(self.create_order_wait)
                if response.status_code == 200:
                    self.error_counter = 0
                    return response.json()
                self.error_handler()
            except requests.exceptions.RequestException as e:
                self.error_handler()

    def create_order_list(self, order_list): # list of orders
        req = {
            "id": 14,
            "method": "private/create-order-list",
            "api_key": self.api_key,
            "params": {
                "contingency_type": "LIST",
                "order_list": order_list
            },
            "nonce": int(time.time() * 1000)    
        }

        param_str = ""
        MAX_LEVEL = 3

        if "params" in req:
            param_str = self.params_to_str(req['params'], 0)

        payload_str = req['method'] + str(req['id']) + req['api_key'] + param_str + str(req['nonce'])

        req['sig'] = hmac.new(
            bytes(str(self.api_secret), 'utf-8'),
            msg=bytes(payload_str, 'utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

        while self.error_counter < 9:
            try:
                response = requests.post("https://api.crypto.com/v2/private/create-order-list", json=req, headers={'Content-Type':'application/json'})
                time.sleep(self.create_order_list_wait)
                if response.status_code == 200:
                    self.error_counter = 0
                    return response.json()
                self.error_handler()
            except requests.exceptions.RequestException as e:
                self.error_handler()

    def get_order_history(self): # CONFIRMS WORKS
        req = {
            "id": 1,
            "nonce" : int(time.time() * 1000),
            "method": "private/get-order-history",
            "api_key": self.api_key,
            "params": {}
        }

        paramString = ""

        if "params" in req:
            for key in req['params']:
                paramString += key
                paramString += str(req['params'][key])

        sigPayload = req['method'] + str(req['id']) + req['api_key'] + paramString + str(req['nonce'])

        req['sig'] = hmac.new(
            bytes(str(self.api_secret), 'utf-8'),
            msg=bytes(sigPayload, 'utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

        while self.error_counter < 9:
            try:
                response = requests.post('https://api.crypto.com/v2/private/get-order-history', json=req)
                time.sleep(self.get_order_history_wait)
                if response.status_code == 200:
                    self.error_counter = 0
                    return response.json()
                self.error_handler()
            except requests.exceptions.RequestException as e:
                self.error_handler()

    def get_instruments(self):
        while self.error_counter < 9:
            try:
                response = requests.get(f'https://api.crypto.com/v2/public/get-instruments')
                time.sleep(self.get_instruments_wait)
                if response.status_code == 200:
                    self.error_counter = 0
                    return response.json()
                self.error_handler()
            except requests.exceptions.RequestException as e:
                self.error_handler()


api = API("U4ue4Wvfo8E7ckdC1XoHu9", "C66tu72RRATpKTb8z7Gs4n")
api.get_candlestick("BTC", -1, -1, -1)