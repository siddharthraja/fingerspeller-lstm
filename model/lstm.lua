require 'torch'
require 'nn'

local LSTM, parent = torch.class('nn.LSTM', 'nn.AbstractRecurrent')

function LSTM:__init(inputSize, outputSize, rho, cell2gate)
    parent.__init(self, rho or 9999)
    self.inputSize = inputSize
    self.outputSize = outputSize
    
    self.cell2gate = (cell2gate == nil) and true or cell2gate
    self.recurrentModule = self:buildModel()

    self.modules[1] = self.recurrentModule
    self.sharedClones[1] = self.recurrentModule

    self.zeroTesor = torch.Tensor()

    self.cells = {}
    self.gradCells = {}

end

function LSTM:buildGate()

    local gate = nn.Sequential()

    if not self.cell2gate then
        gate:add(nn.NarrowTable(1,2))
    end

    local input2gate = nn.Linear(self.inputSize, self.outputSize)
    local output2gate = nn.LinearNoBias(self.outputSize, self.outp


end
