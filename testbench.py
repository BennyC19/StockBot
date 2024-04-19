import time
import datetime
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from main import Agent
from resources.api_keys import crypto_dotcom_key, crypto_dotcom_secret

class Testbench():
    def __init__(self):
        self.agent = Agent(crypto_dotcom_key, crypto_dotcom_secret)
        self.training_data_log = json.load(open("resources/training_data_log.json"))
        self.test_results = json.load(open("test_results/test_results.json"))
    
    def update_test_results_file(self):
        with open("test_results/test_results.json", "w") as file:
            json.dump(self.test_results, file, separators=(',', ':'))
    
    
    """
    Test 1: Test to predict time it will take to train model
    Returns: integer for hours
    """
    def test1_predict_testing_duration(self):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.agent.transformer.parameters(), lr=0.001)
        self.agent.transformer.train()

        start_time = time.time()

        prices_data = torch.rand(100, 1, 20480)
        labels_data = torch.rand(100, 1, 3)

        dataset = TensorDataset(prices_data, labels_data)
        dataloader = DataLoader(dataset, batch_size=self.agent.batch_size, shuffle=False)

        for epoch in range(3):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.agent.device)
                labels = labels.to(self.agent.device)
                output = self.agent.transformer(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                

        self.agent.transformer.eval()
        
        end_time = time.time()
        
        estimated_training_time = int(((start_time - end_time) / 3600) * 10163)

        return estimated_training_time

    """
    Test 2: Retrieves the historical prices using the crypto.com api

    """
    def test2_setup_historical_prices_data(self):

        self.training_data_log["coins"][0]["prices"] = []

        time_now = int(time.time() * 1000)
        time_now = time_now - (time_now % 60000)

        training_time = 63_072_000_000  # two years
        testing_time = 5_184_000_000    # two months

        start_time = time_now - training_time - testing_time
        end_time = start_time + 18000000

        while (end_time <= time_now):
            candlesticks = self.agent.api.get_candlestick("BTC_USD", "1m", start_time, end_time)

            for candle in candlesticks["result"]["data"]:
                self.training_data_log["coins"][0]["prices"].append(float(candle["c"]))

            start_time += 18000000
            end_time += 18000000
                                                                                                                                        # 1036800 + 60 * 1440
        self.training_data_log["coins"][0]["prices"] = self.training_data_log["coins"][0]["prices"][len(self.training_data_log["coins"][0]["prices"]) - 1123200:] # deletes data that is more than 2years and 2 months

        with open("resources/training_data_log.json", "w") as file:
            json.dump(self.training_data_log, file, separators=(',',':'))


    """
    Test 3: Algorithm which analyzes the prices and sets a 'marker' at each price point to determine if the bot should buy, sell, or hold.

    """
    def test3_setup_training_test_data(self, minimum_profit_from_individual_trade):

        self.training_data_log["coins"][0]["training_data"] = []

        for i in range(1, len(self.training_data_log["coins"][0]["prices"]) - 1):
            if (self.training_data_log["coins"][0]["prices"][i] == 0.0):
                self.training_data_log["coins"][0]["prices"][i] = (self.training_data_log["coins"][0]["prices"][i - 1] + self.training_data_log["coins"][0]["prices"][i + 1]) / 2

            if ((self.training_data_log["coins"][0]["prices"][i - 1] > self.training_data_log["coins"][0]["prices"][i]) & (self.training_data_log["coins"][0]["prices"][i + 1] > self.training_data_log["coins"][0]["prices"][i])):
                self.training_data_log["coins"][0]["training_data"].append("B")

            elif ((self.training_data_log["coins"][0]["prices"][i - 1] < self.training_data_log["coins"][0]["prices"][i]) & (self.training_data_log["coins"][0]["prices"][i + 1] < self.training_data_log["coins"][0]["prices"][i])):
                self.training_data_log["coins"][0]["training_data"].append("S")

            else:
                self.training_data_log["coins"][0]["training_data"].append("H")

        last_buy_index = None
        for i in range(len(self.training_data_log["coins"][0]["training_data"])):
            if (self.training_data_log["coins"][0]["training_data"][i] == "S"):
                if (last_buy_index == None):
                    self.training_data_log["coins"][0]["training_data"][i] = "H"
                else:
                    difference = ((self.training_data_log["coins"][0]["prices"][i + 1] / self.training_data_log["coins"][0]["prices"][last_buy_index + 1]) - 1) * 100
                    if (difference <= 0.075 + minimum_profit_from_individual_trade):
                        self.training_data_log["coins"][0]["training_data"][i] = "H"
            elif (self.training_data_log["coins"][0]["training_data"][i] == "B"):
                last_buy_index = i

        last_buy_index = None
        for i in range(len(self.training_data_log["coins"][0]["training_data"])):
            if (self.training_data_log["coins"][0]["training_data"][i] == "B"):
                if (last_buy_index == None):
                    last_buy_index = i
                else:
                    if (self.training_data_log["coins"][0]["prices"][i + 1] < self.training_data_log["coins"][0]["prices"][last_buy_index + 1]):
                        self.training_data_log["coins"][0]["training_data"][last_buy_index] = "H"
                        last_buy_index = i
                    else:
                        self.training_data_log["coins"][0]["training_data"][i] = "H"

            elif (self.training_data_log["coins"][0]["training_data"][i] == "S"):
                last_buy_index = None

        with open("resources/training_data_log.json", "w") as file:
            json.dump(self.training_data_log, file, separators=(',',':'))   


    """
    Test 4: This test offers a graph to analyze the performance of the placement of the markers by the algorithm in test 2

    """
    def test4_verify_training_data(self):

        markers = self.training_data_log["coins"][0]["training_data"]
        x_values = range(len(self.training_data_log["coins"][0]["prices"]) - 2)
        y_values = self.training_data_log["coins"][0]["prices"][1:-1]

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


    """
    Test 5: This test is only used with test 7. This test trains over a period of 100 minutes.

    """
    def test5_train_model_last_hundred(self, end_index):

        training_period = 100
        new_prices_dict = {}
        for coin in self.training_data_log["coins"]:
            new_prices_dict[coin["coin"]] = []
            for i in range(end_index-1-20480-training_period, end_index):
                price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
                new_prices_dict[coin["coin"]].append(price_change)

            price = new_prices_dict[coin["coin"]][:-1] 
            new_prices_dict[coin["coin"]] = self.agent.sigmoid(torch.mul(torch.tensor(price), 100)).tolist()
            
        self.agent.transformer.load_state_dict(torch.load("recources/transformer_model.pth"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.agent.transformer.parameters(), lr=0.001)
        self.agent.transformer.train()

        start_index = 0
        end_index = 20480
        
        training_inputs = []
        training_outputs = []

        while (end_index < 20580): # 1036798 is the number of prices in the log
            
            training_input = []
            training_output = []
            
            for coin in self.training_data_log["coins"]:
                
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

        dataloader = DataLoader(dataset, batch_size=self.agent.batch_size, shuffle=False)

        for epoch in range(3):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.agent.device)
                labels = labels.to(self.agent.device)
                output = self.agent.transformer(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                

        self.agent.transformer.eval()


    """
    Test 6: Will train the model with two years worth of data

    """
    def test6_train_model(self):

        # Step 1: Normalize price data
        new_prices_dict = {}
        for coin in self.training_data_log["coins"]:
            new_prices_dict[coin["coin"]] = []
            for i in range(1, 1036800):
                price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
                new_prices_dict[coin["coin"]].append(price_change)

            price = new_prices_dict[coin["coin"]][:-1] 
            new_prices_dict[coin["coin"]] = self.agent.sigmoid(torch.mul(torch.tensor(price), 100)).tolist()

        # Step 2: Prepare Dataset
        self.agent.transformer.load_state_dict(torch.load("recources/transformer_model.pth"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.agent.transformer.parameters(), lr=0.001)
        self.agent.transformer.train()

        start_index = 0
        end_index = 20480

        next_dataset_list = list(range(20580, 1036798, 100)) # chunks of 100
        next_dataset_list.append(1036798)

        for new_index in next_dataset_list:

            training_inputs = []
            training_outputs = []

            while (end_index < new_index):

                training_input = []
                training_output = []

                for coin in self.training_data_log["coins"]:

                    training_input.append(new_prices_dict[coin["coin"]][start_index:end_index])

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

            dataloader = DataLoader(dataset, batch_size=self.agent.batch_size, shuffle=False)
            for epoch in range(3):
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.agent.device)
                    labels = labels.to(self.agent.device)
                    output = self.agent.transformer(inputs)
                    loss = criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        torch.save(self.agent.transformer.state_dict(), "recources/transformer_model.pth")

        self.agent.transformer.eval()


    """
    Test 7: Will go through the 2 months of prices without retraining
    - Plot a graph of when the bot decided to buy and sell
    - This test will not calculate the potential gains
    """
    def test7_analyze_success_rate_no_retraining(self):

        self.agent.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))
        buy_sell_history = []
        transformer_input = []
        start_index = 1016318
        end_index = 1036798
        
        while (end_index < len(self.training_data_log["coins"][0]["prices"])):
            for coin in self.training_data_log["coins"]:
                transformer_input.append(coin["prices"][start_index:end_index])

            input = torch.tensor(transformer_input).unsqueeze(1).to('cpu')
            
            with torch.no_grad():
                output = self.agent.transformer(input)
        
            output = output.tolist() # takes the tensor and turns it back into a list
            
            for choice, coin in zip(output[0], self.training_data_log["coins"]):
                if ((choice[0] > choice[1]) & (choice[0] > choice[2])): # if Buy is larger
                    buy_sell_history.append("red")

                elif ((choice[1] > choice[0]) & (choice[1] > choice[2])): # if Sell is larger
                    buy_sell_history.append("green")
                
                else: # if hold is larger
                    buy_sell_history.append("grey")
            
            start_index += 1
            end_index += 1
        
        buy_list = []
        gain_history = []
        total_gain = 0
        for price, marker in zip(self.training_data_log["coins"][0]["prices"][1036798:], buy_sell_history):
            if (marker == "red"):
                buy_list.append(price)
            elif (marker == "green"):
                if (len(buy_list) != 0):
                    for price_bought in buy_list:
                        total_gain += ((price / price_bought - 1) * 100) - 0.15
                    buy_list = []

            gain_history.append(total_gain)

        #print(len(buy_sell_history)) <-- 86402
        x_values = range(len(self.training_data_log["coins"][0]["prices"][1036798:]))
        y_values = self.training_data_log["coins"][0]["prices"][1036798:]

        """
        plt.scatter(x_values, y_values, color=buy_sell_history)
        plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
        plt.xlabel("time")
        plt.ylabel("price")
        plt.title("Price with buy and sell choices")
        plt.show()
        """

        x_values = range(len(gain_history))
        y_values = gain_history

        return y_values

        """
        plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
        plt.xlabel("time")
        plt.ylabel("Gain")
        plt.title("Gain over time")
        plt.show() 
        """


    """
    Test 8:
    - Will go through the 2 months of prices with training on new data at regular intervals so that it can learn from new data continuously
    - Plot the success rate of the bot over time
    - Plot potential gains of the bot over the 2 month period
    """
    def test8_analyze_success_rate_regular_retraining(self):

        self.agent.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))
        
        buy_sell_history = []
        transformer_input = []
        start_index = 1016318
        end_index = 1036798
        
        counter = 0
        while (end_index < len(self.training_data_log["coins"][0]["prices"])):

            if (counter == 100):
                self.test5_train_model_last_hundred(end_index)
                counter = 0

            for coin in self.training_data_log["coins"]:
                transformer_input.append(coin["prices"][start_index:end_index])

            input = torch.tensor(transformer_input).unsqueeze(1).to('cpu')
            
            with torch.no_grad():
                output = self.agent.transformer(input)
        
            output = output.tolist() # takes the tensor and turns it back into a list
            
            for choice, coin in zip(output[0], self.training_data_log["coins"]):
                if ((choice[0] > choice[1]) & (choice[0] > choice[2])): # if Buy is larger
                    buy_sell_history.append("red")

                elif ((choice[1] > choice[0]) & (choice[1] > choice[2])): # if Sell is larger
                    buy_sell_history.append("green")
                
                else: # if hold is larger
                    buy_sell_history.append("grey")

            counter += 1
            start_index += 1
            end_index += 1
        
        buy_list = []
        gain_history = []
        total_gain = 0
        for price, marker in zip(self.training_data_log["coins"][0]["prices"][1036798:], buy_sell_history):
            if (marker == "red"):
                buy_list.append(price)
            elif (marker == "green"):
                if (len(buy_list) != 0):
                    for price_bought in buy_list:
                        total_gain += ((price / price_bought - 1) * 100) - 0.15
                    buy_list = []

            gain_history.append(total_gain)

        #print(len(buy_sell_history)) <-- 86402
        x_values = range(len(self.training_data_log["coins"][0]["prices"][1036798:]))
        y_values = self.training_data_log["coins"][0]["prices"][1036798:]

        """
        plt.scatter(x_values, y_values, color=buy_sell_history)
        plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
        plt.xlabel("time")
        plt.ylabel("price")
        plt.title("Price with buy and sell choices")
        plt.show()
        """

        x_values = range(len(gain_history))
        y_values = gain_history

        with open("test_results/gain_history_test3.pkl", "wb") as file:
            pickle.dump(y_values, file)
        
        """
        plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
        plt.xlabel("time")
        plt.ylabel("Gain")
        plt.title("Gain over time")
        plt.show() 
        """


"""
Sequence 1

"""
testbench = Testbench() 
print(f"Predicted training duration for one sequence = {testbench.test1_predict_testing_duration()} hours")
print("//////////////////////////////////////////////////////////////////////////////////")
print("Sequence 1 START")
print(f"[{datetime.datetime.now()}]  started setup_training_test_data (step 1 of 12)")
testbench.test3_setup_training_test_data(0.005)
print(f"[{datetime.datetime.now()}]  finished setup_training_test_data")
print(f"[{datetime.datetime.now()}]  started train_model (step 2 of 12)")
testbench.test6_train_model()
print(f"[{datetime.datetime.now()}]  finished train_model")
print(f"[{datetime.datetime.now()}]  started analyze_success_rate_no_retraining (step 3 of 12)")
result = testbench.test7_analyze_success_rate_no_retraining()
testbench.test_results["sequence_1"]["success_rate_no_retraining"] = result
testbench.update_test_results_file()
print(f"[{datetime.datetime.now()}]  finished analyze_success_rate_no_retraining")
print(f"[{datetime.datetime.now()}]  started analyze_success_rate_regular_retraining (step 4 of 12)")
result = testbench.test8_analyze_success_rate_regular_retraining()
testbench.test_results["sequence_1"]["success_rate_regular_retraining"] = result
testbench.update_test_results_file()
print(f"[{datetime.datetime.now()}]  finished analyze_success_rate_regular_retraining")
print("Sequence 1 END")

"""
Sequence 2

"""

testbench = Testbench() 
print("//////////////////////////////////////////////////////////////////////////////////")
print("Sequence 2 START")
print(f"[{datetime.datetime.now()}]  started setup_training_test_data (step 5 of 12)")
testbench.test3_setup_training_test_data(0.01)
print(f"[{datetime.datetime.now()}]  finished setup_training_test_data")
print(f"[{datetime.datetime.now()}]  started train_model (step 5 of 12)")
testbench.test6_train_model()
print(f"[{datetime.datetime.now()}]  finished train_model")
print(f"[{datetime.datetime.now()}]  started analyze_success_rate_no_retraining (step 7 of 12)")
result = testbench.test7_analyze_success_rate_no_retraining()
testbench.test_results["sequence_2"]["success_rate_no_retraining"] = result
testbench.update_test_results_file()
print(f"[{datetime.datetime.now()}]  finished analyze_success_rate_no_retraining")
print(f"[{datetime.datetime.now()}]  started analyze_success_rate_regular_retraining (step 8 of 12)")
result = testbench.test8_analyze_success_rate_regular_retraining()
testbench.test_results["sequence_2"]["success_rate_regular_retraining"] = result
testbench.update_test_results_file()
print(f"[{datetime.datetime.now()}]  finished analyze_success_rate_regular_retraining")
print("Sequence 2 END")

"""
Sequence 3

"""
testbench = Testbench() 
print("//////////////////////////////////////////////////////////////////////////////////")
print("Sequence 3 START")
print(f"[{datetime.datetime.now()}]  started setup_training_test_data (step 9 of 12)")
testbench.test3_setup_training_test_data(0.02)
print(f"[{datetime.datetime.now()}]  finished setup_training_test_data")
print(f"[{datetime.datetime.now()}]  started train_model (step 10 of 12)")
testbench.test6_train_model()
print(f"[{datetime.datetime.now()}]  finished train_model")
print(f"[{datetime.datetime.now()}]  started analyze_success_rate_no_retraining (step 11 of 12)")
result = testbench.test7_analyze_success_rate_no_retraining()
testbench.test_results["sequence_3"]["success_rate_no_retraining"] = result
testbench.update_test_results_file()
print(f"[{datetime.datetime.now()}]  finished analyze_success_rate_no_retraining")
print(f"[{datetime.datetime.now()}]  started analyze_success_rate_regular_retraining (step 12 of 12)")
result = testbench.test8_analyze_success_rate_regular_retraining()
testbench.test_results["sequence_3"]["success_rate_regular_retraining"] = result
testbench.update_test_results_file()
print(f"[{datetime.datetime.now()}]  finished analyze_success_rate_regular_retraining")
print("Sequence 3 END")

