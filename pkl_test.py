import pickle
with open('rqc_example.pkl', 'rb') as file:
    data = pickle.load(file)

with open('rqc_example.txt', 'w') as out_file:
    print(data, file=out_file)