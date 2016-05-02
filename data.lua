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
    local sequences = torch.Tensor(n_images, channels, width, height)
    local labels = torch.Tensor(n_images)
    local lines = lines_from(datafile)

    --FIXME this assumes a sequence length of 5, edit the input formatting as needed
    for k, v in pairs(lines) do
        local image_1, image_2, image_3, image_4, image_5, class = v:match("([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)")
        image1 = image.scale(image.load(image_1, 3), width, height)
        -- load sequence 
        -- sequence = torch.cat(image1, image2,3):cat(image3,3):cat(image4,3):cat(image5,3)
        sequences[k] =image1 -- sequence
        
        labels[k] = tonumber(class)

    end

    return {sequences, labels}

end


--TODO: change number of image sequences you'd like to load below
--
local n_images = 150
local seq_length = 5
local data, labels = load_data('train.txt', n_images, channels, width, height)

return {
    data = data,
    labels = labels,
    size = n_images
}
