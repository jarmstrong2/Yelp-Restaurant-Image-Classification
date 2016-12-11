require 'torch'
dataSet = torch.load('train_dataset.t7')
dataSet.class = {}

for i = 1, #dataSet.labels do
	label = dataSet.labels[i]
	classTensor = torch.zeros(1,10)
	classStr = ''
	for j = 1, #dataSet.labels[i] do
		str = label:sub(j,j)
		num = tonumber(str)
		if num ~= nil then
			classTensor[{{1},{num + 1}}] = 1
		end
	end
	table.insert(dataSet.class, classTensor)
end

torch.save('train_dataset_class.t7', dataSet)