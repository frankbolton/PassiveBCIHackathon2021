import function_Evaluate_SKtimeBasics

#n_estimators
A = [10,50,100,200,300,400]
i=0

for a in A:
    print(f"{i}: {a} ")
    print(function_Evaluate_SKtimeBasics.runModel(a))
    i=i+1