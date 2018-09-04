#### Paper_Experiment_Predictions ####
# Run on a file and compute accuracy on a nFold cross validation
# Compute also the brier score and the logscore

#Methods and scores to test
doSXGPC = false#Sparse XGPC (sparsity)
doSVGPC = true #Sparse Variational GPC (Hensmann)
doEPGPC = false

PWD=pwd()
include("functions_paper_experiment.jl")
# ExperimentName = "Prediction"
ExperimentName = "Convergence"
@enum ExperimentType AccuracyExp=1 ConvergenceExp=2
ExperimentTypes = Dict("Convergence"=>ConvergenceExp,"Accuracy"=>AccuracyExp)
Experiment = ExperimentTypes[ExperimentName]
doTime = true #Return time needed for training
doAccuracy = true #Return Accuracy
doBrierScore = true # Return BrierScore
doLogScore = true #Return LogScore
doAUCScore = true
doLikelihoodScore = true
doWrite = true #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold
doAutotuning = false
doPointOptimization = false
#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
dataset = "Electricity"
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = 30000 #Maximum number of iterations for every algorithm
iter_points = [1:1:99;100:10:999;1000:100:9999;10000:1000:100000] #Iteration points where measures are taken
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold
MList = [16,32,64,128]
for m in MList
    println("Working now with $m inducing points")
#Main Parameters
global main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["γ"] = 1e-3
main_param["M"] = m #Number of inducing points
main_param["Kernel"] = "rbf"
main_param["theta"] = 3.0 #initial Hyperparameter of the kernel
main_param["BatchSize"] = 100
main_param["Verbose"] = 2
main_param["Window"] = 10
main_param["Autotuning"] = doAutotuning
main_param["PointOptimization"] = doPointOptimization
#BSVM and SVGPC Parameters
SXGPCParam = XGPCParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
SVGPCParam = SVGPCParameters(Stochastic=true,Sparse=true,main_param=main_param)
EPGPCParam = EPGPCParameters(main_param=main_param)

#Global variables for debugging
X = []; y = []; X_test = []; y_test = [];

#Set of all models
global TestModels = Dict{String,TestingModel}()

if doSXGPC; TestModels["SXGPC"] = TestingModel("SXGPC",DatasetName,ExperimentName,"SXGPC",SXGPCParam); end;
if doSVGPC;   TestModels["SVGPC"]   = TestingModel("SVGPC",DatasetName,ExperimentName,"SVGPC",SVGPCParam);      end;
if doEPGPC; TestModels["EPGPC"] = TestingModel("EPGPC",DatasetName,ExperimentName,"EPGPC",EPGPCParam); end;

writing_order = Array{String,1}();                    if doTime; push!(writing_order,"time"); end;
if doAccuracy; push!(writing_order,"accuracy"); end;  if doBrierScore; push!(writing_order,"brierscore"); end;
if doLogScore; push!(writing_order,"-logscore"); end;  if doAUCScore; push!(writing_order,"AUCscore"); end;
if doLikelihoodScore; push!(writing_order,"medianlikelihoodscore"); push!(writing_order,"meanlikelihoodscore"); end;
for (name,testmodel) in TestModels
  println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
  #Initialize the results storage
  testmodel.Model = Array{Any}(undef,nFold)
  testmodel.Results["Time"] = Array{Any}(undef,nFold);
  testmodel.Results["Accuracy"] = Array{Any}(undef,nFold);
  testmodel.Results["MeanL"] = Array{Any}(undef,nFold);
  testmodel.Results["MedianL"] = Array{Any}(undef,nFold);
  testmodel.Results["ELBO"] = Array{Any}(undef,nFold);
  for i in 1:nFold #Run over all folds of the data
    if ShowIntResults
      println("#### Fold number $i/$nFold ###")
    end

    X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
    y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
    if (length(y_test) > 100000 )
        X_test = X_test[StatsBase.sample(1:length(y_test),100000,replace=false),:];
        y_test = y_test[StatsBase.sample(1:length(y_test),100000,replace=false)];
    end
    X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
    y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
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
  end
  if testmodel.MethodName == "SVGPC"
      testmodel.Param["Kernel"] = gpflow.kernels[:Sum]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["theta"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
  end
  ProcessResultsConvergence(testmodel,nFold)
  if doWrite
    top_fold = "data_M$(main_param["M"])";
    if !isdir(top_fold); mkdir(top_fold); end;
    WriteResults(testmodel,top_fold,writing_order) #Write the results in an adapted format into a folder
  end
end#Loop on models
end#Loop on #inducing points
