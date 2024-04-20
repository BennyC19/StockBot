
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from main import Agent
from resources.api_keys import crypto_dotcom_key, crypto_dotcom_secret

agent = Agent(crypto_dotcom_key, crypto_dotcom_secret)

def setup_historical_prices_test_data():

    training_data_log = json.load(open("resources/training_data_log.json"))

    training_data_log["coins"][0]["prices"] = []

    time_now = int(time.time() * 1000)
    time_now = time_now - (time_now % 60000)

    training_time = 63_072_000_000  # two years
    testing_time = 5_184_000_000    # two months

    start_time = time_now - training_time - testing_time
    end_time = start_time + 18000000

    while (end_time <= time_now):
        candlesticks = agent.api.get_candlestick("BTC_USD", "1m", start_time, end_time)

        for candle in candlesticks["result"]["data"]:
            training_data_log["coins"][0]["prices"].append(float(candle["c"]))

        start_time += 18000000
        end_time += 18000000
                                                                                                                                     # 1036800 + 60 * 1440
    training_data_log["coins"][0]["prices"] = training_data_log["coins"][0]["prices"][len(training_data_log["coins"][0]["prices"]) - 1123200:] # deletes data that is more than 2years and 2 months

    with open("resources/training_data_log.json", "w") as file:
        json.dump(training_data_log, file, separators=(',',':'))

def setup_training_test_data(minimum_profit_from_individual_trade):

    training_data_log = json.load(open("resources/training_data_log.json"))

    training_data_log["coins"][0]["training_data"] = []

    for i in range(1, len(training_data_log["coins"][0]["prices"]) - 1):
        if (training_data_log["coins"][0]["prices"][i] == 0.0):
            training_data_log["coins"][0]["prices"][i] = (training_data_log["coins"][0]["prices"][i - 1] + training_data_log["coins"][0]["prices"][i + 1]) / 2

        if ((training_data_log["coins"][0]["prices"][i - 1] > training_data_log["coins"][0]["prices"][i]) & (training_data_log["coins"][0]["prices"][i + 1] > training_data_log["coins"][0]["prices"][i])):
            training_data_log["coins"][0]["training_data"].append("B")

        elif ((training_data_log["coins"][0]["prices"][i - 1] < training_data_log["coins"][0]["prices"][i]) & (training_data_log["coins"][0]["prices"][i + 1] < training_data_log["coins"][0]["prices"][i])):
            training_data_log["coins"][0]["training_data"].append("S")

        else:
            training_data_log["coins"][0]["training_data"].append("H")

    last_buy_index = None
    for i in range(len(training_data_log["coins"][0]["training_data"])):
        if (training_data_log["coins"][0]["training_data"][i] == "S"):
            if (last_buy_index == None):
                training_data_log["coins"][0]["training_data"][i] = "H"
            else:
                difference = ((training_data_log["coins"][0]["prices"][i + 1] / training_data_log["coins"][0]["prices"][last_buy_index + 1]) - 1) * 100
                if (difference <= 0.075 + minimum_profit_from_individual_trade):
                    training_data_log["coins"][0]["training_data"][i] = "H"
        elif (training_data_log["coins"][0]["training_data"][i] == "B"):
            last_buy_index = i

    last_buy_index = None
    for i in range(len(training_data_log["coins"][0]["training_data"])):
        if (training_data_log["coins"][0]["training_data"][i] == "B"):
            if (last_buy_index == None):
                last_buy_index = i
            else:
                if (training_data_log["coins"][0]["prices"][i + 1] < training_data_log["coins"][0]["prices"][last_buy_index + 1]):
                    training_data_log["coins"][0]["training_data"][last_buy_index] = "H"
                    last_buy_index = i
                else:
                    training_data_log["coins"][0]["training_data"][i] = "H"

        elif (training_data_log["coins"][0]["training_data"][i] == "S"):
            last_buy_index = None

    with open("resources/training_data_log.json", "w") as file:
        json.dump(training_data_log, file, separators=(',',':'))   

def verify_training_data():
    training_data_log = json.load(open("resources/training_data_log.json"))
    markers = training_data_log["coins"][0]["training_data"]
    x_values = range(len(training_data_log["coins"][0]["prices"]) - 2)
    y_values = training_data_log["coins"][0]["prices"][1:-1]

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

def test1_train_model_last_hundred(end_index): # This function is correct for the final product but for testing purposes it must be modified

    training_data_log = json.load(open("resources/training_data_log.json"))
    training_period = 100
    new_prices_dict = {}
    for coin in training_data_log["coins"]:
        new_prices_dict[coin["coin"]] = []
        for i in range(end_index-1-20480-training_period, end_index):
            price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
            new_prices_dict[coin["coin"]].append(price_change)

        price = new_prices_dict[coin["coin"]][:-1] 
        new_prices_dict[coin["coin"]] = agent.sigmoid(torch.mul(torch.tensor(price), 100)).tolist()
        
    agent.transformer.load_state_dict(torch.load("recources/transformer_model.pth"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.transformer.parameters(), lr=0.001)
    agent.transformer.train()

    start_index = 0
    end_index = 20480
    
    training_inputs = []
    training_outputs = []

    while (end_index < 20580): # 1036798 is the number of prices in the log
        
        training_input = []
        training_output = []
        
        for coin in training_data_log["coins"]:
            
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

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for epoch in range(3):
        for inputs, labels in dataloader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            output = agent.transformer(inputs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                
    
    torch.save(agent.transformer.state_dict(), "resources/transformer_model.pth")

    agent.transformer.eval()

def test1_train_model(batch_size): # Only train on two years of data

    # Step 1: Normalize price data
    training_data_log = json.load(open("resources/training_data_log.json"))

    new_prices_dict = {}
    for coin in training_data_log["coins"]:
        new_prices_dict[coin["coin"]] = []
        for i in range(1, 1036800):
            price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
            new_prices_dict[coin["coin"]].append(price_change)

        price = new_prices_dict[coin["coin"]][:-1] 
        new_prices_dict[coin["coin"]] = agent.sigmoid(torch.mul(torch.tensor(price), 100)).tolist()

    # Step 2: Prepare Dataset
    agent.transformer.load_state_dict(torch.load("recources/transformer_model.pth"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.transformer.parameters(), lr=0.001)
    agent.transformer.train()

    start_index = 0
    end_index = 20480

    next_dataset_list = [25660, 30840, 36020, 41200, 46380, 51560, 56740, 61920, 67100, 72280, 77460, 82640, 87820, 93000, 98180, 103360, 108540, 113720, 118900, 124080, 129260, 134440, 139620, 144800, 149980, 155160, 160340, 165520, 170700, 175880, 181060, 186240, 191420, 196600, 201780, 206960, 212140, 217320, 222500, 227680, 232860, 238040, 243220, 248400, 253580, 258760, 263940, 269120, 274300, 279480, 284660, 289840, 295020, 300200, 305380, 310560, 315740, 320920, 326100, 331280, 336460, 341640, 346820, 352000, 357180, 362360, 367540, 372720, 377900, 383080, 388260, 393440, 398620, 403800, 408980, 414160, 419340, 424520, 429700, 434880, 440060, 445240, 450420, 455600, 460780, 465960, 471140, 476320, 481500, 486680, 491860, 497040, 502220, 507400, 512580, 517760, 522940, 528120, 533300, 538480, 543660, 548840, 554020, 559200, 564380, 569560, 574740, 579920, 585100, 590280, 595460, 600640, 605820, 611000, 616180, 621360, 626540, 631720, 636900, 642080, 647260, 652440, 657620, 662800, 667980, 673160, 678340, 683520, 688700, 693880, 699060, 704240, 709420, 714600, 719780, 724960, 730140, 735320, 740500, 745680, 750860, 756040, 761220, 766400, 771580, 776760, 781940, 787120, 792300, 797480, 802660, 807840, 813020, 818200, 823380, 828560, 833740, 838920, 844100, 849280, 854460, 859640, 864820, 870000, 875180, 880360, 885540, 890720, 895900, 901080, 906260, 911440, 916620, 921800, 926980, 932160, 937340, 942520, 947700, 952880, 958060, 963240, 968420, 973600, 978780, 983960, 989140, 994320, 999500, 1004680, 1009860, 1015040, 1020220, 1025400, 1030580, 1035760, 1036798] # ++5180

    for new_index in next_dataset_list:
        training_inputs = []
        training_outputs = []

        while (end_index < new_index):

            training_input = []
            training_output = []

            for coin in training_data_log["coins"]:

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

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for epoch in range(1):
            for inputs, labels in dataloader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                output = agent.transformer(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    torch.save(agent.transformer.state_dict(), "recources/transformer_model.pth")

    agent.transformer.eval()

"""
Test 2: Will go through the 2 months of prices without retraining
- Plot a graph of when the bot decided to buy and sell
- This test will not calculate the potential gains
"""
def test2_analyze_success_rate_no_retraining():
    agent.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))
    training_data_log = json.load(open("resources/training_data_log.json"))
    buy_sell_history = []
    transformer_input = []
    start_index = 1016318
    end_index = 1036798
    
    while (end_index < len(training_data_log["coins"][0]["prices"])):
        for coin in training_data_log["coins"]:
            transformer_input.append(coin["prices"][start_index:end_index])

        input = torch.tensor(transformer_input).unsqueeze(1).to('cpu')
        
        with torch.no_grad():
            output = agent.transformer(input)
    
        output = output.tolist() # takes the tensor and turns it back into a list
        
        for choice, coin in zip(output[0], training_data_log["coins"]):
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
    for price, marker in zip(training_data_log["coins"][0]["prices"][1036798:], buy_sell_history):
        if (marker == "red"):
            buy_list.append(price)
        elif (marker == "green"):
            if (len(buy_list) != 0):
                for price_bought in buy_list:
                    total_gain += ((price / price_bought - 1) * 100) - 0.15
                buy_list = []

        gain_history.append(total_gain)

    #print(len(buy_sell_history)) <-- 86402
    x_values = range(len(training_data_log["coins"][0]["prices"][1036798:]))
    y_values = training_data_log["coins"][0]["prices"][1036798:]

    plt.scatter(x_values, y_values, color=buy_sell_history)
    plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
    plt.xlabel("time")
    plt.ylabel("price")
    plt.title("Price with buy and sell choices")
    plt.show()

    x_values = range(len(gain_history))
    y_values = gain_history

    plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
    plt.xlabel("time")
    plt.ylabel("Gain")
    plt.title("Gain over time")
    plt.show() 

"""
Test 3:
- Will go through the 2 months of prices with training on new data at regular intervals so that it can learn from new data continuously
- Plot the success rate of the bot over time
- Plot potential gains of the bot over the 2 month period
"""
def test3_analyze_success_rate_regular_retraining():
    agent.transformer.load_state_dict(torch.load("resources/transformer_model.pth"))
    training_data_log = json.load(open("resources/training_data_log.json"))
    buy_sell_history = []
    transformer_input = []
    start_index = 1016318
    end_index = 1036798
    
    counter = 0
    while (end_index < len(training_data_log["coins"][0]["prices"])):

        if (counter == 0):
            test1_train_model_last_hundred(end_index)
            counter = 0

        for coin in training_data_log["coins"]:
            transformer_input.append(coin["prices"][start_index:end_index])

        input = torch.tensor(transformer_input).unsqueeze(1).to('cpu')
        
        with torch.no_grad():
            output = agent.transformer(input)
    
        output = output.tolist() # takes the tensor and turns it back into a list
        
        for choice, coin in zip(output[0], training_data_log["coins"]):
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
    for price, marker in zip(training_data_log["coins"][0]["prices"][1036798:], buy_sell_history):
        if (marker == "red"):
            buy_list.append(price)
        elif (marker == "green"):
            if (len(buy_list) != 0):
                for price_bought in buy_list:
                    total_gain += ((price / price_bought - 1) * 100) - 0.15
                buy_list = []

        gain_history.append(total_gain)

    #print(len(buy_sell_history)) <-- 86402
    x_values = range(len(training_data_log["coins"][0]["prices"][1036798:]))
    y_values = training_data_log["coins"][0]["prices"][1036798:]

    plt.scatter(x_values, y_values, color=buy_sell_history)
    plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
    plt.xlabel("time")
    plt.ylabel("price")
    plt.title("Price with buy and sell choices")
    plt.show()

    x_values = range(len(gain_history))
    y_values = gain_history

    plt.plot(x_values, y_values, color="black", linestyle="-", linewidth=1)  # Connect points with a black line
    plt.xlabel("time")
    plt.ylabel("Gain")
    plt.title("Gain over time")
    plt.show() 
