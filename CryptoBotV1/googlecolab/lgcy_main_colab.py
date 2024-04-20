from googlecolab.transformer_colab import Transformer
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/content/googlecolab')

data_log = json.load(open("/content/googlecolab/data_log_colab.json"))
sigmoid = nn.Sigmoid()
transformer = Transformer().to('cuda')

"""
Training the Transformer with all the data
"""
def test1(): 
    new_prices_dict = {}
    for coin in data_log["coins"]:
        new_prices_dict[coin["coin"]] = []
        for i in range(1, len(coin["prices"])):
            price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
            new_prices_dict[coin["coin"]].append(price_change)

        price = new_prices_dict[coin["coin"]][:-1] 
        new_prices_dict[coin["coin"]] = sigmoid(torch.mul(torch.tensor(price), 100)).tolist()

    transformer.load_state_dict(torch.load("/content/googlecolab/transformer_model_colab.pth"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)
    transformer.train()

    num_epochs = 3
    for epoch in range(num_epochs):
        start_index = 0
        end_index = 20480
        end_index_counter = 0
        while (end_index <= 1036798):

            training_inputs = []
            training_outputs = []
        
            for coin in data_log["coins"]:
                training_inputs.append(new_prices_dict[coin["coin"]][start_index:end_index])
                if (coin["training_data"][end_index] == "B"):
                    training_outputs.append([1., 0., 0.])
                elif (coin["training_data"][end_index] == "S"):
                    training_outputs.append([0., 1., 0.])
                else:
                    training_outputs.append([0., 0., 1.])
            
            input = torch.tensor(training_inputs).unsqueeze(1).to('cuda')
            label= torch.tensor(training_outputs).to('cuda')

            output = transformer(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            start_index += 1
            end_index += 1

    torch.save(transformer.state_dict(), "/content/googlecolab/transformer_model_colab.pth")

    transformer.eval()

def test2():
    new_prices_dict = {}
    for coin in data_log["coins"]:
        new_prices_dict[coin["coin"]] = []
        for i in range(1, len(coin["prices"])):
            price_change = (coin["prices"][i] - coin["prices"][i-1]) / coin["prices"][i-1]
            new_prices_dict[coin["coin"]].append(price_change)

        price = new_prices_dict[coin["coin"]][:-1]
        new_prices_dict[coin["coin"]] = sigmoid(torch.mul(torch.tensor(price), 100)).tolist()

    transformer.load_state_dict(torch.load("/content/googlecolab/transformer_model_colab.pth"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)
    transformer.train()

    start_index = 0
    end_index = 20480

    next_dataset_list = [25660, 30840, 36020, 41200, 46380, 51560, 56740, 61920, 67100, 72280, 77460, 82640, 87820, 93000, 98180, 103360, 108540, 113720, 118900, 124080, 129260, 134440, 139620, 144800, 149980, 155160, 160340, 165520, 170700, 175880, 181060, 186240, 191420, 196600, 201780, 206960, 212140, 217320, 222500, 227680, 232860, 238040, 243220, 248400, 253580, 258760, 263940, 269120, 274300, 279480, 284660, 289840, 295020, 300200, 305380, 310560, 315740, 320920, 326100, 331280, 336460, 341640, 346820, 352000, 357180, 362360, 367540, 372720, 377900, 383080, 388260, 393440, 398620, 403800, 408980, 414160, 419340, 424520, 429700, 434880, 440060, 445240, 450420, 455600, 460780, 465960, 471140, 476320, 481500, 486680, 491860, 497040, 502220, 507400, 512580, 517760, 522940, 528120, 533300, 538480, 543660, 548840, 554020, 559200, 564380, 569560, 574740, 579920, 585100, 590280, 595460, 600640, 605820, 611000, 616180, 621360, 626540, 631720, 636900, 642080, 647260, 652440, 657620, 662800, 667980, 673160, 678340, 683520, 688700, 693880, 699060, 704240, 709420, 714600, 719780, 724960, 730140, 735320, 740500, 745680, 750860, 756040, 761220, 766400, 771580, 776760, 781940, 787120, 792300, 797480, 802660, 807840, 813020, 818200, 823380, 828560, 833740, 838920, 844100, 849280, 854460, 859640, 864820, 870000, 875180, 880360, 885540, 890720, 895900, 901080, 906260, 911440, 916620, 921800, 926980, 932160, 937340, 942520, 947700, 952880, 958060, 963240, 968420, 973600, 978780, 983960, 989140, 994320, 999500, 1004680, 1009860, 1015040, 1020220, 1025400, 1030580, 1035760, 1036798] # ++5180

    for new_index in next_dataset_list:
        training_inputs = []
        training_outputs = []

        while (end_index < new_index):

            training_input = []
            training_output = []

            for coin in data_log["coins"]:

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

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        for epoch in range(1):
            counter = 0
            for inputs, labels in dataloader:
                counter += 1
                if (counter == 20):
                    print("bruh")
                    counter = 0
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                output = transformer(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    torch.save(transformer.state_dict(), "/content/googlecolab/transformer_model_colab.pth")

    transformer.eval()
