require 'torch'
require 'image'
require 'nnx'
require 'rnn'
require 'optim'

-- load data module
--

local data = require 'data'

--hyper-parameters
--
local lstm = nn.Sequential()
local lr   = 1e-3 -- learning rate
local lstm_input_size = 4608 -- size of final conv output
local batch_size = 4
local rho = 1 --for 1:1 comparison to frame by frame
local hidden_size = 100
local epochs = 100
local train_size = 150
local n_classes = 15 -- for letters, change for lipreading

local learning_rate = .001

local criterion = nn.ClassNLLCriterion()

local sequence_model = nn:Sequential()
:add(nn.SpatialConvolutionMM(3, 16, 5, 5))
:add(nn.ReLU())
:add(nn.SpatialMaxPooling(4,4))
:add(nn.SpatialConvolutionMM(16,32, 7, 7))
:add(nn.ReLU())
:add(nn.SpatialMaxPooling(4, 4))
:add(nn.Reshape(4608))


local feedback_module = nn.Linear(1, rho)
local transfer = nn.Sigmoid()

local stepmodule = nn.Sequential()
local r = nn.FastLSTM(lstm_input_size, n_classes)
lstm:add(nn.LookupTable(rho, hidden_size))
lstm:add(nn.Sequencer(r))
lstm:add(nn.SelectTable(-1))

sequence_model:add(nn.Sequential():add(nn.Linear(lstm_input_size, n_classes)):add(nn.LogSoftMax()))

local train_data = data.data[1]
local train_labels = data.data[2]
local test_data = data.data[3]
local test_labels = data.data[4]

local indices = torch.LongTensor(batch_size)
local data, labels = torch.Tensor(batch_size), torch.Tensor(batch_size)

local total_valid = 0
local total_counted =0

for iteration=1, epochs do
    indices:random(1, batch_size)
    data:index(train_data, 1, indices)
    labels:index(train_labels, 1, indices)

    sequence_model:zeroGradParameters()

    local outputs = sequence_model:forward(data)
    local err = criterion:forward(outputs, labels)

    local outstr = string.format("NLL err= %f", err)

    print(outstr)
    local temp =0
    local argmax =0
    for i= 1, batch_size do
        temp, argmax = torch.max(outputs[i], 1)
        if argmax[1] == labels[i] then
            total_valid = total_valid + 1.0
        end
        total_counted = total_counted + 1.0
    end
    
    print(string.format('mean class accuracy (train): %f', total_valid/total_counted * 100))

    local gradOutputs = criterion:backward(outputs, labels)
    local gradInputs = sequence_model:backward(data, gradOutputs)

    sequence_model:updateParameters(lr)
end

