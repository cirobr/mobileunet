"""
LibML.Learn!()
Author: cirobr@GitHub
Date: 08-Jul-2024
"""

@info "Project start"
cd(@__DIR__)

### arguments
# cudadevice = parse(Int64, ARGS[1])
# nepochs    = parse(Int64, ARGS[2])
# debugflag  = parse(Bool,  ARGS[3])

cudadevice = 0
nepochs    = 200
debugflag  = false

script_name = basename(@__FILE__)
@info "script_name: $script_name"
@info "cudadevice: $cudadevice"
@info "nepochs: $nepochs"
@info "debugflag: $debugflag"


### libs
using Pkg
envpath = expanduser("~/envs/debug/")
Pkg.activate(envpath)

using CUDA
CUDA.device!(cudadevice)
CUDA.versioninfo()

using Flux
import Flux: relu, leakyrelu
using Images
using DataFrames
using CSV
using JLD2
using FLoops
using Random
using Statistics: mean, minimum, maximum, norm
using StatsBase: sample
using MLUtils: splitobs, kfolds, obsview, ObsView

# private libs
using TinyMachines; const tm=TinyMachines
using PreprocessingImages; const p=PreprocessingImages
using PascalVocTools; const pv=PascalVocTools
using LibML
import LibML: IoU_loss, ce3_loss, cosine_loss, softloss
using LibCUDA
# include("../architectures.jl")

LibCUDA.cleangpu()
@info "environment OK"


### constants
const KB = 1024
const MB = KB * KB
const GB = KB * MB


### folders
outputfolder = script_name[1:end-3] * "/"

# pwd(), homedir()
workpath = pwd() * "/"
workpath = replace(workpath, homedir() => "~")
datasetpath = "./dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist

modelspath  = workpath * "models/" * outputfolder
mkpath(expanduser(modelspath))

tblogspath  = workpath * "tblogs/" * outputfolder
rm(tblogspath; force=true, recursive=true); sleep(1)   # sleep to ensure removal
mkpath(expanduser(tblogspath))
@info "folders OK"


### datasets
@info "creating datasets..."
classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [pv.voc_classname2classnumber[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-coi-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
dftrain = dftrain[dftrain.segmented .== 1,:]

fpfn = expanduser(datasetpath) * "dfvalid-coi-resized.csv"
dfvalid = CSV.read(fpfn, DataFrame)
dfvalid = dfvalid[dfvalid.segmented .== 1,:]


########### debug ############
if debugflag
      dftrain = first(dftrain, 5)
      dfvalid = first(dfvalid, 2)
      minibatchsize = 1
      epochs  = 2
else
      minibatchsize = 1
      epochs  = nepochs
end
##############################


train_images = []
valid_images = []
train_masks  = []
valid_masks  = []
dfs = [dftrain, dfvalid]
Xs  = [train_images, valid_images]
ys  = [train_masks, valid_masks]

@floop for (df, X, y) in zip(dfs, Xs, ys)
      N = size(df, 1)
      for i in 1:N   # no @floop here
            local fpfn = expanduser(df.X[i])
            img = Images.load(fpfn)
            img = p.color2Float32(img)
            push!(X, img)

            local fpfn = expanduser(df.y[i])
            local mask = Images.load(fpfn)
            local mask = pv.voc_rgb2classes(mask)
            local mask = Flux.onehotbatch(mask, [0,classnumbers...],0)
            local mask = permutedims(mask, (2,3,1))
            push!(y, mask)
      end
end

Xtrain = cat(train_images...; dims=4); train_images=nothing
ytrain = cat(train_masks...; dims=4);  train_masks=nothing
Xvalid = cat(valid_images...; dims=4); valid_images=nothing
yvalid = cat(valid_masks...; dims=4);  valid_masks=nothing
@info "tensors OK"

# dataloaders
Random.seed!(1234)   # to enforce reproducibility
trainset = Flux.DataLoader((Xtrain, ytrain),
                            batchsize=minibatchsize,
                            shuffle=true) |> gpu
validset = Flux.DataLoader((Xvalid, yvalid),
                            batchsize=1,
                            shuffle=false) |> gpu
@info "dataloader OK"


# check memory requirements
Xtr = Xtrain[:,:,:,1:1] |> gpu
ytr = ytrain[:,:,:,1:1] |> gpu

dpsize = sizeof(Xtr) + sizeof(ytr)
dpGB = dpsize / GB
@info "datapoints in trainset = $(size(Xtrain,4))"
@info "datapoints in 1 GB = $(1 / dpGB)"


LibCUDA.cleangpu()


### model
Random.seed!(1234)   # to enforce reproducibility
modelcpu = tm.MobileUNet(3,C; verbose=false)
# fpfn = expanduser("")
# LibML.loadModelState!(fpfn, modelcpu)
model    = modelcpu |> gpu;
@info "model OK"


### check for matching between model and data
@assert size(model(Xtr)) == size(ytr) || error("model/data features do not match")
@info "model/data matching OK"
###


# loss functions
lossFunction(yhat, y) = LibML.IoU_loss(yhat, y)
lossfns = [lossFunction]
@info "loss functions OK"


# optimizer
optimizerFunction = Flux.Adam
η = 1e-4
λ = 0.0      # default 5e-4
modelOptimizer = λ > 0 ? Flux.Optimiser(WeightDecay(λ), optimizerFunction(η)) : optimizerFunction(η)


optimizerState = Flux.setup(modelOptimizer, model)
# Flux.freeze!(optimizerState.enc)
@info "optimizer OK"


### training
@info "start training ..."

number_since_best = 10
patience = 5
metrics = [
      LibML.AccScore,
      LibML.F1Score,
      LibML.IoUScore,
      # Flux.mse,
      # LibML.ce3_loss,
]

Random.seed!(1234)   # to enforce reproducibility
LibML.Learn!(epochs, model, (trainset, validset), optimizerState, lossfns;
      metrics=metrics,
      earlystops=(number_since_best, patience),
      modelspath=modelspath,
      tblogspath=tblogspath
)

# rename model -> bestmodel
fpfn = expanduser(modelspath) * "model.jld2"
mv(fpfn, expanduser(modelspath) * "bestmodel.jld2", force=true)


LibCUDA.cleangpu()
@info "project finished"
