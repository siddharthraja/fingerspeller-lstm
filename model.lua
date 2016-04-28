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
local lstm_input_size = 4096 -- size of final conv output
local batch_size = 8
local rho = 3 --sequence length
local hidden_size = 100
local epochs = 100
local train_size = 100
local n_classes = 26 -- for letters, change for lipreading

local learning_rate = .001

local criterion = nn.ClassNLLCriterion()

local input_model = nn:Sequential()
:add(nn.SpatialConvolutionMM(15, 16, 5, 5))
:add(nn.ReLU())
:add(nn.SpatialMaxPooling(4,4))
:add(nn.SpatialConvolutionMM(16,32, 7, 7))
:add(nn.ReLU())
:add(nn.SpatialMaxPooling(4, 4))
:add(nn.Reshape(4096))


local feedback_module = nn.Linear(1, rho)
local transfer = nn.Sigmoid()

local stepmodule = nn.Sequential()
local r = nn.FastLSTM(lstm_input_size, n_classes)

local lstm = nn.Sequential()
lstm:add(nn.LookupTable(rho, hidden_size))
lstm:add(nn:SplitTable(1,2))
lstm:add(nn.Sequencer(r))
lstm:add(nn.SelectTable(-1))
lstm:add(nn.Linear(hidden_size, n_classes))
lstm:add(nn.LogSoftMax())

stepmodule:add(lstm)
input_model:add(nn.Sequencer(stepmodule))

local train_data = data.data
local train_labels = data.labels

local train_logger = optim.Logger('train.log')
local indices = torch.LongTensor(batch_size)
local data, labels = torch.Tensor(batch_size), torch.Tensor(batch_size)

for iteration=1, epochs do
    indices:random(1, batch_size)
    data:index(train_data, 1, indices)
    labels:index(train_labels, 1, indices)

    input_model:zeroGradParameters()

    local outputs = input_model:forward(inputs)
    local err = criterion:forward(outputs, labels)

    local outstr = string.format("NLL err= %f", err)
    train_logger:add(outstr)
    print(outstr)

    local gradOutputs = criterion:backward(outputs, labels)
    local gradInputs = input_model:backward(inputs, gradOutputs)

    input_model:updateParameters(lr)
end

