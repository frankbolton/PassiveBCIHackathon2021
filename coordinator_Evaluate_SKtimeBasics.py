import function_Evaluate_SKtimeBasics

#n_estimators
A = [10, 20,40,50,100,200]
#narrow or don't narrow the EEG chanel space
B = [True,False]
i=0

for a in A:
    for b in B:
        print(f"{i}: {a} {b}")
        print(function_Evaluate_SKtimeBasics.runModel(a,b))
        i=i+1