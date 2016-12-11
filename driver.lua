
local cmd = torch.CmdLine()

cmd:text()
cmd:text('Script for training model.')

cmd:option('-hiddenSize', 500, 'number of hidden units in lstms')
cmd:option('-lr' , 1e-3, 'learning rate')
cmd:option('-batchSize' , 5, 'mini batch size')
cmd:option('-iterations' , 10, 'number of iterations of training')
cmd:option('-numelems' , 100, 'number of elements of training')

cmd:text()
opt = cmd:parse(arg)

dofile('model.lua')
dofile('train.lua')