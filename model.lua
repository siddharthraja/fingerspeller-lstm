require 'torch'
require 'image'
require 'nnx'
require 'rnn'
require 'optim'

-- load data module
--

local data, labels, size = require 'data'

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

local input_model = nn:Sequential()
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

local lstn = nn.Sequential()
lstm:add(nn.LookupTable(rho, hidden_size))
lstm:add(nn.Sequencer(r))
lstm:add(nn.SelectTable(-1))
input_model:add(lstn:add(nn.Linear(lstm_input_size, n_classes)):add(nn.LogSoftMax()))


local train_data = data.data[1]
local train_labels = data.data[2]

local train_logger = optim.Logger('train.log')
local indices = torch.LongTensor(batch_size)
local data, labels = torch.Tensor(batch_size), torch.Tensor(batch_size)

for iteration=1, epochs do
    indices:random(1, batch_size)
    data:index(train_data, 1, indices)
    labels:index(train_labels, 1, indices)

    input_model:zeroGradParameters()

    local outputs = input_model:forward(data)
    local err = criterion:forward(outputs, labels)

    local outstr = string.format("NLL err= %f", err)
    train_logger:add{ ["NLL err "]=err}
    print(outstr)

    local gradOutputs = criterion:backward(outputs, labels)
    local gradInputs = input_model:backward(data, gradOutputs)

    input_model:updateParameters(lr)
end

