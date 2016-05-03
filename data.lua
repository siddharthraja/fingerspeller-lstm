require 'torch'
require 'image'
require 'nnx'

-- data parameters
--
local width = 224
local height= 224
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
    local lines = lines_from(datafile)
    local sequences = torch.Tensor(#lines, channels, width, height)
    local labels = torch.Tensor(#lines)
    --FIXME this assumes a sequence length of 5, edit the input formatting as needed
    for k, v in pairs(lines) do
        local image_1, image_2, image_3, image_4, image_5, class = v:match("([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)")
        image1 = image.scale(image.load(image_1, 3), width, height)
        image2 = image.scale(image.load(image_2, 3), width, height)
        image3 = image.scale(image.load(image_3, 3), width, height)
        image4 = image.scale(image.load(image_4, 3), width, height)
        image5 = image.scale(image.load(image_5, 3), width, height)

        -- load sequence 
        images = torch.cat(image1, image2,1):cat(image3,1):cat(image4,1):cat(image5,1)
        local seq = images[{{1,3},{},{}}] --squash sequence down
        sequences[k] = seq
        
        labels[k] = tonumber(class)

    end

    return {sequences, labels}

end


--TODO: change number of image sequences you'd like to load below
--
local n_images = 150
local seq_length = 5
local data, labels = load_data('train.txt', n_images, channels, width, height)
local test_data, test_labels = load_data('test.txt', 50, 3, width, height)

return {
    data = data,
    labels = labels,
    test_data = test_data,
    test_labels = test_labels
}
