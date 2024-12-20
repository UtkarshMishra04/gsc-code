import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import pickle

class DecisionDiffuserDataset(Dataset):

    def __init__(self, dataset_path, mode="diffusion", selected_skill=None):
        self.dataset_path = dataset_path

        self.mode = mode
        self.selected_skill = selected_skill

        assert self.mode in ["diffusion", "inverse"]
        if self.mode == "inverse":
            assert self.selected_skill is not None, "Please specify the skill to be used for inverse model training"

        self.transition_data = []
        self.inverse_obsdata = {}
        self.inverse_actions = {}
        self.inverse_objorder = {}

        task_skeleton = "pick place pick place pick place"
        same_keys = [[0, 2, 4], [1, 3, 5]]
        
        for file in os.listdir(self.dataset_path):
            if file.endswith(".pkl"):
                with open(os.path.join(self.dataset_path, file), 'rb') as f:
                    data = pickle.load(f)
                    obs = data['obs']
                    obs = np.concatenate(obs, axis=0)
                    init_obj_order = np.arange(8).reshape(1, -1)
                    obj_order = np.array(data['obj_order'])
                    obj_order = np.concatenate([init_obj_order, obj_order], axis=0).reshape(-1, 1)
                    final_obs = np.concatenate([obs, obj_order], axis=1)
                    data['obs'] = final_obs
                    self.transition_data.append(data['obs'])

                    for i in range(len(data['action'])):
                        if i not in self.inverse_obsdata.keys():
                            self.inverse_obsdata[i] = []
                            self.inverse_actions[i] = []
                            self.inverse_objorder[i] = []
                        self.inverse_obsdata[i].append(data['obs'][8*i:8*(i+2)][:, :12])
                        self.inverse_actions[i].append(data['action'][i])
                        self.inverse_objorder[i].append(data['obs'][8*(i+1):8*(i+2)][:, 12:].reshape(-1))

        for k in self.inverse_obsdata.keys():
            self.inverse_obsdata[k] = np.array(self.inverse_obsdata[k])
            self.inverse_actions[k] = np.array(self.inverse_actions[k])
            self.inverse_objorder[k] = np.array(self.inverse_objorder[k])

        for sk in same_keys:
            for i in range(1, len(sk)):
                self.inverse_obsdata[sk[0]] = np.concatenate([self.inverse_obsdata[sk[0]], self.inverse_obsdata[sk[i]]], axis=0)
                self.inverse_actions[sk[0]] = np.concatenate([self.inverse_actions[sk[0]], self.inverse_actions[sk[i]]], axis=0)
                self.inverse_objorder[sk[0]] = np.concatenate([self.inverse_objorder[sk[0]], self.inverse_objorder[sk[i]]], axis=0)

                self.inverse_obsdata.pop(sk[i])
                self.inverse_actions.pop(sk[i])
                self.inverse_objorder.pop(sk[i])

    def __len__(self):
        return len(self.transition_data)

    def __getitem__(self, idx):
        if self.mode == "diffusion":
            return self.transition_data[idx]
        else:
            if self.selected_skill == "pick":
                return self.inverse_obsdata[0][idx], self.inverse_actions[0][idx], self.inverse_objorder[0][idx]
            elif self.selected_skill == "place":
                return self.inverse_obsdata[1][idx], self.inverse_actions[1][idx], self.inverse_objorder[1][idx]
            else:
                raise NotImplementedError

def main():

    dataset_path = "../dd_dataset/task1/traj_data"

    dataset1 = DecisionDiffuserDataset(dataset_path, mode="diffusion")
    dataset2 = DecisionDiffuserDataset(dataset_path, mode="inverse", selected_skill="pick")
    dataset3 = DecisionDiffuserDataset(dataset_path, mode="inverse", selected_skill="place")

    print(len(dataset1))
    print(len(dataset2))
    print(len(dataset3))

    sample1 = dataset1[0]
    sample2 = dataset2[0]
    sample3 = dataset3[0]

    print(sample1.shape)
    print(sample2[0].shape, sample2[1].shape, sample2[2].shape)
    print(sample3[0].shape, sample3[1].shape, sample3[2].shape)

if __name__ == "__main__":
    main()