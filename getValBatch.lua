function getValBatch(data, batchSize, restCount, fileCount)
	classObj = data.class
	fileObj = data.files

	restSize = #fileObj

	-- prepare for input/target data
	targettensor = torch.zeros(batchSize, 10)--:cuda()
	inputtensor = torch.zeros(batchSize, 1008)--:cuda()

	for i = 1, batchSize do

		dataSize = #fileObj[restCount]

		if fileCount > dataSize then
			fileCount = 1
			restCount = restCount + 1
			if restCount > restSize then
				restCount = 1
			end
		end

		fileId = fileObj[restCount][fileCount]
		filePath = "train_tensors/" .. fileId .. ".t7"
		fileTensor = torch.load(filePath)

		inputtensor[{{i},{}}] = fileTensor--:cuda()

		targetCurrent = classObj[restCount]
		compareZeroTarget = torch.eq(targetCurrent, torch.zeros(1,10))
		resultCompareZeroTarget = compareZeroTarget:prod()

		firstBin = torch.Tensor(1,1):fill(resultCompareZeroTarget)
		targetCurrent = torch.cat(firstBin, targetCurrent[{{},{1,9}}], 2)

		targettensor[{{i},{}}] = targetCurrent--:cuda()

		fileCount = fileCount + 1
	end

	return inputtensor, targettensor, restCount, fileCount

end