require 'torch'
require 'image'
require 'nnx'
require 'rnn'

--TODO still a work in progress

-- data parameters
--

local width = 64
local height= 64
local channels = 3


local function file_exists(file)

    local f = io.open(file, 'rb')
    if f then f:close() else print(file + " not found") end
    return f ~= nil
end

local function lines_from(file)
    if not file_exists(file) then return {} end
    lines = {}
    for line in io.lines(file) do
        lines[#lines + 1] = line
    end
    return lines
end


local function load_data(datafile, n_images, channels, width, height)
    local sequences = torch.Tensor(n_images, 3, channels, width, height)
    local labels = torch.Tensor(n_images)
    local lines = lines_from(datafile)

    --FIXME this assumes a sequence length of 3 btw
    for k, v in pairs(lines) do
        local image_1, image_2, image_3, class = v:match("([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)")
        sequences[k][1] = image.scale(image.load(image_1, channels), width, height)
        sequences[k][2] = image.scale(image.load(image_2, channels), width, height)
        sequences[k][3] = image.scale(image.load(image_3, channels), width, height)
        labels[k] = tonumber(class) + 1

    end

    return {sequences, labels}

end



--hyper-parameters
--

local batchsize = 8
local rho = 3 --sequence length
local hidden_size = 100
local epochs = 100
local train_size = 100
local n_classes = 26 -- for letters, change for lipreading

local learning_rate = .001

local criterion = nn.ClassNLLCriterion()

--TODO this is just a few linear layers for now, replace with convolutional
local input_model = nn:Sequential()
:add(nn.Reshape(width*height*channels))
:add(nn.Linear(4096, 4096))
:add(nn.Linear(4096, n_classes))

local feedback_module = nn.Linear(1, rho)
local transfer = nn.Sigmoid()


local r = nn.Recurrent(hidden_size,
                input_model,
                feedback_module,
                transfer,
                rho
                )


local rnn = nn.Sequential()

rnn:add(nn.LookupTable(rho, hidden_size))
rnn:add(nn:SplitTable(1,2))
rnn:add(nn.Sequencer(r))
rnn:add(nn.SelectTable(-1))
rnn:add(nn.Linear(hidden_size, n_classes))
rnn:add(nn.LogSoftMax())

local train_data, train_labels = load_data('train.txt')

local train_logger = optim.Logger('train.log')
local indices = torch.LongTensor(batch_size)
local data, labels = torch.Tensor(batch_size), torch.Tensor(batch_size)

for iteration=1, epochs do
    indices:random(1, batch_size)
    data:index(train_data, 1, indices)
    labels:index(train_labels, 1, indices)

    rnn:zeroGradParameters()

    local outputs = rnn:forward(inputs)
    local err = criterion:forward(outputs, labels)

    local outsr = string.format("NLL err= %f", err)
    train_logger:add(outstr)
    print(outstr)

    local gradOutputs = criterion:backward(outputs, labels)
    local gradInputs = rnn:backward(inputs, gradOutputs)

    rnn:updateParameters(lr)
end

