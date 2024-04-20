import os
import schedule
import json
import time
from time import sleep
from datetime import datetime
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from backend.api import API
from backend.transformer import Transformer
from resources.api_keys import crypto_dotcom_key, crypto_dotcom_secret

# startup 
"""
1. Instantiate all classes
2. Fill databank with the latest prices to the nereast 15m interval
3. Fill databank with the latest training data
4. Repeat 1 and 2 again in case it takes more than 15 minutes
5. Update the model with the latest data
6. Update the user information
"""

class Agent():
    def __init__(self, crypto_dotcom_key, crypto_dotcom_secret):

        self.maker_fee = 0.075
        self.taker_fee = 0.075
        self.error_count = 0
        self.api = API(crypto_dotcom_key, crypto_dotcom_secret, shutdown_callback=self.shutdown)
        self.device = 'cuda'

        if not torch.cuda.is_available(): 
            print("no gpu available")
            exit()

        torch.cuda.empty_cache()
        self.transformer = Transformer().to(self.device)
        random_input_tensor = torch.rand(1, 1, 20480)
        random_output_tensor = torch.rand(1, 1, 3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        inputs = random_input_tensor.to(self.device)
        labels = random_output_tensor.to(self.device)
        output = self.transformer(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache() 
    
        self.sigmoid = nn.Sigmoid()
        self.data_log = json.load(open("resources\data_log.json"))

        self.coin_num = len(self.data_log['coins'])
        
        self.batch_size = int(int(15 / (peak_memory / 1024**3)) / self.coin_num)

    def update_data_log_file(self):
        with open("resources\data_log.json", "w") as file:
            json.dump(self.data_log, file, separators=(',', ':'))
    
    def startup(self):

        if (len(self.data_log['coins']) != 0):

            self.load_model_into_transformer()

            self.update_data_log_file()
        
            self.get_latest_prices()

            self.calculate_training_data()
            
            self.train_with_latest_data()

            self.get_latest_prices()

            self.calculate_training_data()
            
            self.train_with_latest_data()

    def shutdown(self):
        exit()

    def list_coins(self):
        for coin in self.data_log["coins"]:
            print(coin["coin"])

    def add_coin(self, coin_name):

        self.data_log["coins"].append(
            {
                "coin": coin_name,
                "amount_held": 0,
                "minimum_quantity": self.find_minimum_quantity(coin_name),
                "last_price_time": 0,
                "last_training_data_time": 0,
                "prices": [],
                "training_data": []
            }
        )
        self.update_data_log_file()

    def delete_coin(self, coin_name):
        index = 0
        for coin in self.data_log["coins"]:
            if (coin["coin"] == coin_name):
                del self.data_log["coins"][index]
            index += 1
        self.update_data_log_file()

    def find_minimum_quantity(self, coin_name):
        instruments = self.api.get_instruments()
        for instrument in instruments["result"]["instruments"]:
            if (f"{coin_name}_USD" == instrument["instrument_name"]):
                return float(instrument["min_quantity"])

    def save_price_training_data_log(self):
        self.price_training_data_log.to_csv("resources/price_training_data_log.csv")

    def load_model_into_transformer(self):
        self.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))

    def get_latest_prices(self):  

        time_now = int(time.time() * 1000)  # Convert to milliseconds
        time_now = time_now - (time_now % 60000)  # Round down to the nearest minute

        for coin in self.data_log["coins"]:
            
            if (coin["last_price_time"] == 0): # if prices have not been retreived yet
                start_time = time_now - 63_072_000_000
                end_time = start_time + 18000000
                
            elif (((time_now - coin["last_price_time"]) / 60000) < 300):
                start_time = coin["last_price_time"]
                end_time = time_now

            else:
                start_time = coin["last_price_time"]
                end_time = start_time + 18000000

            while (end_time <= time_now):
                
                candlesticks = self.api.get_candlestick(f"{coin['coin']}_USD", "1m", start_time, end_time) 
    
                """
                Add prices to csv here
                """
                for candle in candlesticks["result"]["data"]:
                    coin["prices"].append(float(candle["c"])) # prices are strings in the response so it must be turned into a float

                start_time += 18000000
                end_time += 18000000
            
            # Deleting data that is more than 2 years old
            coin["prices"] = coin["prices"][len(coin["prices"]) - 1036800:] # <-- BUG? Why does this number does not add up to 365daysx2

            # Update "last_price_time"
            coin["last_price_time"] = time_now
            
            self.update_data_log_file()
    
    def calculate_training_data(self):
            
        for coin in self.data_log["coins"]:
            coin["training_data"] = []
            """
            Adding the Buy, Sell and Hold Markers on all points.
            """
            for i in range(1, len(coin["prices"]) - 1):
                if (coin["prices"][i] == 0.0): # In case a value is 0
                    coin["prices"][i] = (coin["prices"][i - 1] + coin["prices"][i + 1]) / 2

                if ((coin["prices"][i - 1] > coin["prices"][i]) & (coin["prices"][i + 1] > coin["prices"][i])):
                    # Buy
                    coin["training_data"].append("B")

                elif ((coin["prices"][i - 1] < coin["prices"][i]) & (coin["prices"][i + 1] < coin["prices"][i])):
                    # Sell
                    coin["training_data"].append("S")

                else:
                    # Hold
                    coin["training_data"].append("H")
            
            """
            If a Sell is not profitable because the difference between the price bought and price sold 
            is not large enough to overcome the taker fee then don"t sell at that spot.
            """
            last_buy_index = None
            for i in range(len(coin["training_data"])):
                if (coin["training_data"][i] == "S"):
                    if (last_buy_index == None):
                        coin["training_data"][i] = "H"
                    else:
                        difference = ((coin["prices"][i + 1] / coin["prices"][last_buy_index + 1]) - 1) * 100
                        if (difference <= self.taker_fee + 0.005):
                            coin["training_data"][i] = "H"
                elif (coin["training_data"][i] == "B"):
                    last_buy_index = i

            """
            Now that the unprofitable Sell markers are gone, there may be several Buy markers in a row but only the most profitable one should remain 
            so the one with the lowest price stays and the others are turned into Hold markers.
            """
            last_buy_index = None
            for i in range(len(coin["training_data"])):
                
                if (coin["training_data"][i] == "B"):
                    if (last_buy_index == None):
                        last_buy_index = i
                    else:
                        if (coin["prices"][i + 1] < coin["prices"][last_buy_index + 1]): # if new Buy price is lower than previous
                            coin["training_data"][last_buy_index] = "H"
                            last_buy_index = i

                        else: # if last Buy price is lower than new 
                            coin["training_data"][i] = "H" # Keep old Buy but new one turns into Hold

                elif (coin["training_data"][i] == "S"):
                    last_buy_index = None

            self.update_data_log_file()

    def train_with_latest_data(self):        
        
        time_now = int(time.time() * 1000)  # Convert to milliseconds
        time_now = time_now - (time_now % 60000)  # Round down to the nearest minute

        new_prices_dict = {}
        for coin in self.data_log["coins"]:    # This is just so that the slicing is only done once            
            new_prices_dict[coin["coin"]] = []
            for i in range(1, len(coin["prices"])):
                price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
                new_prices_dict[coin["coin"]].append(price_change)

            price = new_prices_dict[coin["coin"]][:-1] 
            new_prices_dict[coin["coin"]] = self.sigmoid(torch.mul(torch.tensor(price), 100)).tolist()
            
        self.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        self.transformer.train()

        num_epochs = 1
        for epoch in range(num_epochs):

            if (self.data_log["last_trained"] == 0):
                start_index = 0
                end_index = 20480
            else:
                minutes_passed = int(time_now - self.data_log["last_trained"])
                end_index = 1036798 - minutes_passed
                start_index = end_index - 20480

            while (end_index <= 1036798): # 1036798 is the number of prices in the log # BUG is it tho?

                training_inputs = []
                training_outputs = []

                for coin in self.data_log["coins"]:
                    
                    training_inputs.append(new_prices_dict[coin["coin"]][start_index:end_index]) # appends price changes to training inputs shape = []

                    if (coin["training_data"][end_index] == "B"):  # BUG end_index could become out of range
                        training_outputs.append([1., 0., 0.])
                    elif (coin["training_data"][end_index] == "S"):
                        training_outputs.append([0., 1., 0.])
                    else:
                        training_outputs.append([0., 0., 1.])

                #input = torch.tensor(training_inputs) input of shape [2, 20480] <-- makes it easier to  
                input = torch.tensor(training_inputs).unsqueeze(1)#.to(self.device)
                label = torch.tensor(training_outputs)#.to(self.device)

                output = self.transformer(input)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                start_index += 1
                end_index += 1 

                self.update_data_log_file() # POTENTIAL BUG ??? WHAT IS BEING UPDATED IN THE LOG FILE?

        self.data_log["last_trained"] = time_now
        
        torch.save(self.transformer.state_dict(), "resources/transformer_model.pth")

        self.transformer.eval()

    def train_with_latest_data_v2(self): # trains the last 100 minutes
        training_period = 100
        new_prices_dict = {}
        for coin in self.data_log["coins"]:
            new_prices_dict[coin["coin"]] = []
            for i in range(-1-20480-training_period, 0):
                price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
                new_prices_dict[coin["coin"]].append(price_change)

            price = new_prices_dict[coin["coin"]][:-1] 
            new_prices_dict[coin["coin"]] = self.sigmoid(torch.mul(torch.tensor(price), 100)).tolist()
            
        self.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        self.transformer.train()

        start_index = 0
        end_index = 20480
        
        training_inputs = []
        training_outputs = []

        while (end_index < (20480 + training_period)):
            
            training_input = []
            training_output = []
            
            for coin in self.data_log["coins"]:
                
                training_input.append(new_prices_dict[coin["coin"]][start_index:end_index]) # appends price changes to training inputs shape = []
                
                if (coin["training_data"][end_index] == "B"):
                    training_output.append([1., 0., 0.])
                elif (coin["training_data"][end_index] == "S"):
                    training_output.append([0., 1., 0.])
                else:
                    training_output.append([0., 0., 1.])
            
            training_inputs.append(training_input)
            training_outputs.append(training_output)
            
            start_index += 1
            end_index += 1

        prices_data = torch.tensor(training_inputs)
        labels_data = torch.tensor(training_outputs)
        
        dataset = TensorDataset(prices_data, labels_data)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(3):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output = self.transformer(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                
        
        torch.save(self.transformer.state_dict(), "resources/transformer_model.pth")

        self.transformer.eval()

    def train_with_all_data(self):

        # Step 1: Normalize price data
        new_prices_dict = {}
        for coin in self.data_log["coins"]:    # This is just so that the slicing is only done once            
            new_prices_dict[coin["coin"]] = []
            for i in range(1, len(coin["prices"])):
                price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
                new_prices_dict[coin["coin"]].append(price_change)

            price = new_prices_dict[coin["coin"]][:-1] 
            new_prices_dict[coin["coin"]] = self.sigmoid(torch.mul(torch.tensor(price), 100)).tolist()

        # Step 2: Prepare Dataset
        self.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        self.transformer.train()

        start_index = 0
        end_index = 20480
        
        next_dataset_list = list(range(20580, 1036798, 100)) # chunks of 100
        next_dataset_list.append(1036798) 
        
        for new_index in next_dataset_list: # This is so that the transformer can be trained on segments of the training data at a time instead of the entire thing (166Gb)
            
            training_inputs = []
            training_outputs = []

            while (end_index < new_index): # 1036798 is the number of prices in the log
                
                training_input = []
                training_output = []
                
                for coin in self.data_log["coins"]:
                    
                    training_input.append(new_prices_dict[coin["coin"]][start_index:end_index]) # appends price changes to training inputs shape = []
                    
                    if (coin["training_data"][end_index] == "B"):
                        training_output.append([1., 0., 0.])
                    elif (coin["training_data"][end_index] == "S"):
                        training_output.append([0., 1., 0.])
                    else:
                        training_output.append([0., 0., 1.])
                
                training_inputs.append(training_input)
                training_outputs.append(training_output)
                
                start_index += 1
                end_index += 1

            prices_data = torch.tensor(training_inputs)
            labels_data = torch.tensor(training_outputs)
            
            dataset = TensorDataset(prices_data, labels_data)

            # Step 3: Put Dataset into dataloader

            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            # Step 4: Train

            for epoch in range(3):
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    output = self.transformer(inputs)
                    loss = criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                
        
        torch.save(self.transformer.state_dict(), "resources/transformer_model.pth")

        self.transformer.eval()


    def bot_analyze(self):
        
        # get prices from data_log and turn them into tensor of right size
        transformer_input = []
        for coin in self.data_log["coins"]:
            transformer_input.append(coin["prices"][-20480:])
        
        input = torch.tensor(transformer_input).unsqueeze(1).to(self.device)
        
        # process through transformer output and create a sorted list by doing the following...
            # Do not include any "Hold" in the list
        
        with torch.no_grad():
            output = self.transformer(input)

        output = output.tolist()
        buy_list = []
        sell_list = []
        for choice, coin in zip(output[0], self.data_log["coins"]):
            if ((choice[0] > choice[1]) & (choice[0] > choice[2])): # if Buy is larger
                buy_list.append([coin["coin"], round((0.05 * round(choice[0] / 0.05)), 2)]) # probability that it should buy in intervals of 0.05

            elif ((choice[1] > choice[0]) & (choice[1] > choice[2])): # if Sell is larger
                sell_list.append([coin["coin"], round((0.05 * round(choice[1] / 0.05)), 2)]) # probability that it should sell in intervals of 0.05

            else: # if hold is larger
                pass
        
        # check how much of each coin is available
        
        user_balance = self.api.get_user_balance()
        
        """
        Algorithm to determine how much of a coin to sell
        """
        for balance in user_balance["result"]["data"][0]["position_balances"]:
            for order in sell_list:
                if (order[0] == balance["instrument_name"]):
                    order[1] = float(balance["quantity"]) * order[1]
        
        self.create_sell_list(sell_list)

        user_balance = self.api.get_user_balance()

        """
        Algorithm to determine how much of a coin to buy
        """
        available_usd = None
        for balance in user_balance["result"]["data"][0]["position_balances"]:
            if (balance["instrument_name"] == "USD"):
                available_usd = float(balance["quantity"])
        
        if (len(buy_list) == 1):
            buy_list[0][1] = buy_list[0][1] * available_usd

        elif (len(buy_list) > 1):
            total = 0
            for order in buy_list:
                total += order[1]

            for order in buy_list:
                order[1] = (order[1] / total) * available_usd
        
        self.create_buy_list(buy_list)

    def create_buy_list(self, buy_list):
        orders = []
        if (len(buy_list) == 0):
            pass 
        else:
            for order in buy_list:
                
                orders.append(
                    {
                        "instrument_name": f"{order[0]}_USD",
                        "side": "BUY",
                        "type": "MARKET",
                        "notional": format(order[1], f".{2}f")
                    }
                )
    
            self.api.create_order_list(orders)

    def create_sell_list(self, sell_list):
        orders = []
        if (len(sell_list) == 0):
            pass 
        else:
            for order in sell_list:

                minimum_quantity_decimal_places = next(item for item in self.data_log["coins"] if item["coin"] == order[0])["minimum_quantity"]

                orders.append(
                    {
                        "instrument_name": f"{order[0]}_USD",
                        "side": "SELL",
                        "type": "MARKET",
                        "quantity": format(order[1], f".{minimum_quantity_decimal_places}f")
                    }
                )
            
            self.api.create_order_list(orders)

    def test_markers(self):
        markers = self.data_log["coins"][0]["training_data"]
        x_values = range(len(self.data_log["coins"][0]["prices"]) - 2)
        y_values = self.data_log["coins"][0]["prices"][1:-1]
        
        colors = []
        for marker in markers:
            if (marker == "B"):
                colors.append("red")
            elif (marker == "S"):
                colors.append("green")
            else:
                colors.append("grey")

        plt.scatter(x_values, y_values, color=colors)
        plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Connected Scatter Plot with Different Colors")
        plt.show()   

#agent = Agent(crypto_dotcom_key, crypto_dotcom_secret)
#torch.save(agent.transformer.state_dict(), "resources/transformer_model.pth")
#torch.save(agent.transformer.state_dict(), "googlecolab/transformer_model_colab.pth")
#agent.get_latest_prices()
#agent.calculate_training_data()
#agent.train_with_latest_data()
#agent.train_with_all_data() # Size of tensor going into transformer when training: [1, 2, 20480]
#agent.bot_analyze() # Size of tensor going into transformer when analyzing:  
#agent.train_with_latest_data_v2()
"""
sleep(60 - datetime.now().second) # to ensure that functions happen exactly on the minute

schedule.every(1).minutes.do(agent.get_latest_prices)
schedule.every(1).minutes.do(agent.bot_analyze)
schedule.every(15).minutes.do(agent.calculate_training_data)
#schedule.every(15).minutes.do(agent.train_with_latest_data)

while True:
    schedule.run_pending()
    sleep(1)
"""