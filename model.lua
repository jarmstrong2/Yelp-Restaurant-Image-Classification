require 'rnn'
require 'nn'
require 'nngraph'

function new_soft_max(mean)
	local lin1 = nn.Linear(opt.hiddenSize, 2000)(mean)
	local relu = nn.ReLU()(lin1)
	local out = nn.Dropout(0.6)(relu)
	local lin2 = nn.Linear(2000, 2)(out)
	return nn.LogSoftMax()(lin2)
end

local input_table = nn.Identity()()
local mask_table = nn.Identity()()
local divisor = nn.Identity()()

local LSTM1 = nn.LSTM(1008, opt.hiddenSize)
local LSTM2 = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
local LSTM3 = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
local LSTMs = nn.Sequential():add(LSTM1):add(LSTM2):add(LSTM3)

rnn = nn.Sequencer(LSTMs)(input_table)

for i = 1, opt.numelems do
	rnn_entry = nn.SelectTable(i)(rnn)
	mask_entry = nn.SelectTable(i)(mask_table)
	result = nn.CMulTable()({rnn_entry,mask_entry})
	if i == 1 then
        join_result = result
    else
        join_result = nn.JoinTable(2)({join_result,result})
    end
end

reshape_join = nn.Reshape(opt.numelems,opt.hiddenSize)(join_result)

join_split = nn.SplitTable(1,2)(reshape_join)

added_join = nn.CAddTable()(join_split)

mean_set = nn.CMulTable()({added_join, divisor})

local soft1 = new_soft_max(mean_set)
local soft2 = new_soft_max(mean_set)
local soft3 = new_soft_max(mean_set)
local soft4 = new_soft_max(mean_set)
local soft5 = new_soft_max(mean_set)
local soft6 = new_soft_max(mean_set)
local soft7 = new_soft_max(mean_set)
local soft8 = new_soft_max(mean_set)
local soft9 = new_soft_max(mean_set)
local soft10 = new_soft_max(mean_set)

mod = nn.gModule({input_table, mask_table, divisor}, 
	{soft1, soft2, soft3, soft4, soft5, 
    	soft6, soft7, soft8, soft9, soft10})

params, grad_params = mod:getParameters()

crit = {}
for i = 1, 10 do
	table.insert(crit, nn.ClassNLLCriterion())
end