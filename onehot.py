import numpy as np
import json
from tqdm import tqdm
import gc
#import cupy

class MakeData():
    def onehot_vec(self, vector, dic_size):
        onehot_vector = np.zeros((25,dic_size+1))
        for i,vec in enumerate(vector):
            onehot_vector[i][vec] = 1
        return onehot_vector
        
    def program_sep(self, program_data):
        programs = []
        labels = []
        for num in range(11):
            for one_program in program_data['{}'.format(num)]:
                one_program_new = []
                one_program_new2 = []
                for token in one_program:
                    if token==9 or token==16 or token==42: # ; { } のとき改行
                        one_program_new2.append(token)
                        one_program_new.append(one_program_new2)
                        one_program_new2 = []
                    else:
                        one_program_new2.append(token)
                programs.append(one_program_new)
                #print(len(programs))
                flag = 0
                if len(programs[-1]) > 100:
                    del programs[-1]
                else:
                    for line in programs[-1]:
                        if len(line) > 25:
                            flag = 1
                        else:
                            pass
                    if flag == 1:
                        del programs[-1]
                    else:
                        labels.append(num)
                #print(len(programs))
                #print(len(line))
                #labels.append(num)
    
        return zip(programs, labels)


    def main(self):
        dataset = {}
        with open('./dataset.json') as jsonfile:
            vector_dict = json.load(jsonfile)
        with open('./data.json') as f:
            token2vec = json.load(f)
        dict_size = len(token2vec)
        count = 0
        batch_size = 10
        programs_and_labels = self.program_sep(vector_dict)
        for programs, labels in programs_and_labels:
            train_x = []
            train_y = []
            for separate_line in programs:
                train_x.append(self.onehot_vec(separate_line,dict_size))
                train_y.append(labels)
        
#    for num in tqdm(range(1,12)):
#        dataset[num] = []
#        for vec in tqdm(vector_dict['{}'.format(num)]):
            #print(num)
            #print(vec)
            #print(len(vec))
#            dataset[num].append(onehot_vec(vec,dict_size))
        #print(dataset)

if __name__=='__main__':
    md = MakeData()
    md.main()
