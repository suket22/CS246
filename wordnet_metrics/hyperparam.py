import linecache
yhat_1 = []
yhat_0 = []
count = 0
with open('duplicate_sample.out') as f1:
    with open('wordnet_true_simvalues.out') as f2:
        for line1 in f1:
            count = count + 1
            if(line1.split()[2] == '1'):
                line2 = linecache.getline('wordnet_false_simvalues.out', count)
                yhat_1.append(line2.split()[2])
            else: yhat_0.append(line2.split()[2])
        
yhat_1.sort()
yhat_0.sort()

print "-----------yhat 1 --------"
for i in range(0,len(yhat_1)):
    print yhat_1[i]
    
print "-----------yhat 0 --------"
for i in range(0,len(yhat_0)):
    print yhat_0[i]