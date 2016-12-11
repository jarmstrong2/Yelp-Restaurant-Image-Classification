function shuffleTable(t)
    local rand = math.random 
    assert( t, "shuffleTable() expected a table, got nil" )
    local iterations = #t
    local j
    
    for i = iterations, 2, -1 do
        j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end

function getBatch(data, count, batchSize)
	classObj = data.class
	fileObj = data.files

	dataSize = #classObj

	-- prepare tables for data as well as mask and divisor
	datatensor = torch.zeros(batchSize,1008)
	masktensor = torch.zeros(batchSize,opt.hiddenSize)
	divisortensor = torch.ones(batchSize, opt.hiddenSize)
	datatable = {}
	masktable = {}
	targettensor = torch.zeros(batchSize, 10)

	for i = 1, opt.numelems do
		table.insert(datatable, datatensor:clone())
		table.insert(masktable, masktensor:clone())
	end

	for i = 1, batchSize do
		timesteps = #fileObj[count]
		if timesteps > opt.numelems then
			timesteps = opt.numelems
		end
		filesList = fileObj[count]
		shuffleTable(filesList)
		targettensor[{{i},{}}] = classObj[count]

		for j = 1, timesteps do
			fileId = filesList[j]
			filePath = "train_tensors/" .. fileId .. ".t7"
			fileTensor = torch.load(filePath)

			datatable[j][{{i},{}}] = fileTensor
			masktable[j][{{i},{}}] = torch.ones(1,opt.hiddenSize)
		end

		divisortensor[{{i},{}}]:fill(1/timesteps)

		count = count + 1
		if count > dataSize then
			count = 1
		end 
	end

	return datatable, masktable, divisortensor, targettensor, count
end
