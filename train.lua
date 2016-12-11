require 'getBatch'
require 'gnuplot'
require 'optim'
require 'nn'

params:uniform(-0.008, 0.008)

traindata = torch.load('train.t7')
valdata = torch.load('val.t7')

traincount = 1
classResult = nil

function classification(results, target)
	guess = torch.zeros(opt.batchSize, 10)

	for i = 1, 10 do
		maxnums, maxinds = torch.max(result[i],2)
		maxinds = maxinds - 1
		guess[{{},{i}}] = maxinds
	end

	print(guess)

	print(target)


	comparison = torch.eq(guess, target)

	rightGuess = comparison:prod(2)

	return rightGuess:sum()/opt.batchSize
end

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    datatable, masktable, divisortensor, targettensor, traincount = 
    getBatch(traindata, traincount, opt.batchSize)

    result = mod:forward({datatable, masktable, divisortensor})

    loss = 0

    critgrad = {}

    for i = 1, 10 do
    	curloss = crit[i]:forward(result[i], targettensor[{{},{i}}]:squeeze() + 1)
    	loss = loss + curloss
    	table.insert(critgrad, 
    		crit[i]:backward(result[i], targettensor[{{},{i}}]:squeeze() + 1))
    end

    classResult = classification(results, targettensor)

    maxnums, maxinds = torch.max(result[1],2)

    mod:backward({datatable, masktable, divisortensor}, critgrad)

    loss = loss/10

    collectgarbage()

    return loss, grad_params

end

local optim_state = {learningRate = opt.lr, alpha = 0.95, epsilon = 1e-8}
local iterations = opt.iterations
local minValLoss = 1/0

for i = 1, iterations do

    local _, loss = optim.adam(feval, params, optim_state)
    print(loss)
    print(classResult)

end
