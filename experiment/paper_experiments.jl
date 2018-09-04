#### Paper_Experiments ####
# Run on a file and compute Accuracy on a nFold cross validation
# Compute also the brier score and the logscore

include("get_arguments.jl")

#Compare XGPC, BSVM, SVGPC and Logistic Regression

#Methods and scores to test
doSXGPC = args["XGPC"] #Sparse XGPC (sparsity)
doEPGPC = args["EPGPC"] #EP GPC
doSVGPC = args["SVGPC"] #Sparse Variational GPC (Hensmann)
doAutotuning = args["autotuning"]
doPointOptimization = args["point-optimization"]

#Add the path necessary modules
PWD = pwd()
include("functions_paper_experiment.jl")

ExperimentName = "Convergence"

doTime = true #Return time needed for training
doAccuracy = true #Return Accuracy
doBrierScore = true # Return BrierScore
doLogScore = true #Return LogScore
doAUCScore = true
doLikelihoodScore = true
doSaveLastState = args["last-state"]
doPlot = args["plot"]
doWrite = !args["no-writing"] #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold


#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
# dataset = "Diabetis"
dataset = args["dataset"]
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = args["maxiter"] #Maximum number of iterations for every algorithm
iter_points= vcat(1:99,100:10:999,1000:100:9999,10000:1000:99999)
(nSamples,nFeatures) = size(X_data);
nFold = args["nFold"]; #Choose the number of folds
iFold = args["iFold"] > nFold ? nFold : args["iFold"]; #Number of fold to estimate
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold

#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["γ"] = 1e-3
main_param["M"] = 100#args["indpoints"]!=0 ? args["indpoints"] : min(100,floor(Int64,0.2*nSamples)) #Number of inducing points
main_param["Kernel"] = "rbf"
main_param["theta"] = 3.0 #initial Hyperparameter of the kernel
main_param["BatchSize"] = args["batchsize"]
main_param["Verbose"] = 2
main_param["Window"] = 10
main_param["Autotuning"] = doAutotuning
main_param["PointOptimization"] = doPointOptimization
#Parameters
SXGPCParam = XGPCParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
SVGPCParam = SVGPCParameters(Stochastic=true,Sparse=true,main_param=main_param)
EPGPCParam = EPGPCParameters(main_param=main_param)

#Set of all models
TestModels = Dict{String,TestingModel}()

if doSXGPC; TestModels["SXGPC"] = TestingModel("SXGPC",DatasetName,ExperimentName,"SXGPC",SXGPCParam); end;
if doSVGPC;   TestModels["SVGPC"]   = TestingModel("SVGPC",DatasetName,ExperimentName,"SVGPC",SVGPCParam);      end;
if doEPGPC; TestModels["EPGPC"] = TestingModel("EPGPC",DatasetName,ExperimentName,"EPGPC",EPGPCParam); end;

#Main printing
print("Dataset $dataset loaded, starting $ExperimentName experiment,")
print(" with $iFold fold out of $nFold, autotuning = $doAutotuning and optindpoints = $doPointOptimization,")
print(" max of $MaxIter iterations\n")
writing_order = ["Time","Accuracy","MeanL","MedianL","ELBO"]
for (name,testmodel) in TestModels
    println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
    #Initialize the results storage
    testmodel.Model = Array{Any}(undef,nFold)
    testmodel.Results["Time"] = Array{Any}(undef,nFold);
    testmodel.Results["Accuracy"] = Array{Any}(undef,nFold);
    testmodel.Results["MeanL"] = Array{Any}(undef,nFold);
    testmodel.Results["MedianL"] = Array{Any}(undef,nFold);
    testmodel.Results["ELBO"] = Array{Any}(undef,nFold);
    for i in 1:iFold #Run over iFold folds of the data
        if ShowIntResults
            println("#### Fold number $i/$nFold ####")
        end
        X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
        y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
        if (length(y_test) > 100000 )#When test set is too big, reduce it for time purposes
            subset = StatsBase.sample(1:length(y_test),100000,replace=false)
            X_test = X_test[subset,:];
            y_test = y_test[subset];
        end
        X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
        y = Float64.(y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))])
        init_t = time_ns()
        CreateModel!(testmodel,i,X,y)
        if testmodel.MethodType == "EPGPC"
            LogArrays = TrainModelwithTime!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)
            testmodel.Results["Time"][i] = LogArrays[1,:]
        else
            LogArrays= hcat(TrainModelwithTime!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)...)
            testmodel.Results["Time"][i] = TreatTime(init_t,LogArrays[1,:],LogArrays[6,:])
        end
        testmodel.Results["Accuracy"][i] = LogArrays[2,:]
        testmodel.Results["MeanL"][i] = LogArrays[3,:]
        testmodel.Results["MedianL"][i] = LogArrays[4,:]
        testmodel.Results["ELBO"][i] = LogArrays[5,:]
        if ShowIntResults
            println("$(testmodel.MethodName) : Time  = $((time_ns()-init_t)*1e-9)s, accuracy : $(LogArrays[2,end])")
        end
        if doWrite && doSaveLastState
            top_fold = PWD*"/results";
            if !isdir(top_fold); mkdir(top_fold); end;
            WriteLastStateParameters(testmodel,top_fold,X_test,y_test,i)
        end
        #Reset the kernel
        if testmodel.MethodName == "SVGPC"
            testmodel.Param["Kernel"] = gpflow.kernels[:Sum]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
        end
    end # of the loop over the folds
    ProcessResultsConvergence(testmodel,iFold)
    if doWrite
        top_fold = "results";
        if !isdir(top_fold); mkdir(top_fold); end;
        WriteResults(testmodel,top_fold,writing_order) #Write the results in an adapted format into a folder
    end
end #Loop over the models
if doPlot
    PlotResultsConvergence(TestModels)
end
