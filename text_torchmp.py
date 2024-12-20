import torch    
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp

import time

class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

def eval(rank, model, device, data, target, result_queue=None):
    model.eval()
    correct = 0
    start = time.time()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        # pred = output.argmax(dim=1, keepdim=True)
        # correct += pred.eq(target.view_as(pred)).sum().item()

    print("Time taken = {} for Rank {}".format(time.time() - start, rank))

    return result_queue.put(output.detach().cpu().clone())

if __name__ == '__main__':
    start = time.time()
    mp.set_start_method('spawn')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(784, 10).to(device)
    model.share_memory()

    data = torch.randn(1000, 784).to(device)
    target = torch.randn(1000).to(device)

    result_queue = mp.Queue()

    processes = []
    for rank in range(4):
        rank_data = data[rank * 250: (rank + 1) * 250]
        rank_target = target[rank * 250: (rank + 1) * 250]
        p = mp.Process(target=eval, args=(rank, model, device, rank_data, rank_target, result_queue))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    results1 = []
    for p in processes:
        temp = result_queue.get()
        results1.append(temp.clone())
        del temp

    results1 = torch.cat(results1, dim=0)

    print("Time taken = {}".format(time.time() - start))

    start = time.time()

    pred = eval(0, model, device, data, target)

    print("Time taken = {}".format(time.time() - start))

    assert torch.all(torch.eq(results1, pred))