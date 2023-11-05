
import pandas as pd 
import numpy as np
data = pd.read_csv('Training_examples.csv')
concept = np.array(data)[:,:-1]
print("The concepts to be learned") 
print(concept)
target = np.array(data)[:,-1] 
print("\nLabels specific to the concepts\n") 
print(target)
print("\n")
def learn(concept,target):
    specific_h = concept[0].copy()
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    for i,h in enumerate(concept): 
        if target[i]=='Yes':
            for j in range(len(specific_h)): 
                if h[j]!=specific_h[j]:
                    specific_h[j]='?' 
                    general_h[j][j]='?'
        elif target[i]=='No':
            for j in range(len(specific_h)):
                if h[j]!=specific_h[j]:
                    general_h[j][j]=specific_h[j] 
                else:
                    general_h[j][j]='?'
    indices = [i for i,val in enumerate(general_h) if val==['?' for i in range(len(specific_h))]] 
    for i in indices:
        general_h.remove(['?' for i in range(len(specific_h))]) 
    return specific_h,general_h
s_final, g_final = learn(concept,target) 
print("Final S: ", s_final)
print("Final G: ", g_final)