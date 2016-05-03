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

sequence_model:add(lstm)
sequence_model.modules[#sequence_model.modules] = nn.Sequential():add(nn.Linear(lstm_input_size, n_classes)):add(nn.LogSoftMax())

local train_data = data.data[1]
local train_labels = data.data[2]
local test_data = data.test_data[1]
local test_labels = data.test_data[2]

local indices = torch.LongTensor(batch_size)
local test_set_size = 14
local test_indices = torch.LongTensor(test_set_size)
local data, labels = torch.Tensor(batch_size), torch.Tensor(batch_size)

local total_valid = 0
local total_counted = 0
local test_batch, test_label_batch = torch.Tensor(batch_size), torch.Tensor(batch_size)

local outplotter = optim.Logger('out.log')
test_acc = 0--(test_acc + (test_counted / test_max))/2.0
for iteration=1, epochs do
    indices:random(1, batch_size)
    data:index(train_data, 1, indices)
    labels:index(train_labels, 1, indices)
    
    test_indices:random(1, 80)
    test_batch:index(test_data, 1, test_indices)
    test_label_batch:index(test_labels, 1, test_indices) 

    sequence_model:zeroGradParameters()

    local outputs = sequence_model:forward(data)
    local err = criterion:forward(outputs, labels)

    local outstr = string.format("NLL err= %f", err)
    local gradOutputs = criterion:backward(outputs, labels)
    local gradInputs = sequence_model:backward(data, gradOutputs)

    sequence_model:updateParameters(lr)
    -- end of training step

    -- Logging and plotting classifier output
    print(outstr)
    local train_acc =0
    local temp =0 
    local argmax =0 for i= 1, batch_size do temp, argmax = torch.max(outputs[i], 1)        if argmax[1] == labels[i] then total_valid = total_valid + 1.0
        end total_counted = total_counted + 1.0 
    train_acc = total_valid / total_counted
    end

    local current_test_acc = 0
    local test_counted = 0
    local test_output = sequence_model:forward(test_batch)
    local test_max = 0

    for i=1, test_set_size do temp, argmax = torch.max(test_output[i], 1)
        if argmax[1] == test_label_batch[i] then test_counted = test_counted + 1.0
        end test_max = test_max + 1.0 
    end
    test_acc = (test_acc + (test_counted / test_max))/2.0
    print(string.format('mean class accuracy (train): %f', train_acc * 100))
    print(string.format('mean class accuracy (test): %f',  test_acc * 100))
    outplotter:add{['Train accuracy']=train_acc, ['Test accuracy']=test_acc}
    outplotter:style{['Train accuracy']='-'}
    outplotter:style{['Test accuracy']='-'}
    outplotter:plot()
end

