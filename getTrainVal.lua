require 'torch'

local cmd = torch.CmdLine()

cmd:text()
cmd:option('-tname' , 'train1.t7')
cmd:option('-vname' , 'val1.t7')

cmd:text()
opt = cmd:parse(arg)

u = torch.randperm(2000)

torch.manualSeed(u[1])
dataSet = torch.load('train_dataset_class.t7')
size_of_set = 2000
size_of_val = 100
classObj = dataSet.class
fileObj = dataSet.file_name

validation = {}
validation.files = {}
validation.class = {}
training = {}
training.files = {}
training.class = {}

for i = 1,100 do
	ind = (torch.ceil(torch.rand(1)*size_of_set)):squeeze()
	validation.files[i] = fileObj[ind]
	validation.class[i] = classObj[ind]
	table.remove(classObj, ind)
	table.remove(fileObj, ind)
	size_of_set = size_of_set - 1
end

training.files = fileObj
training.class = classObj

torch.save(opt.tname,training)
torch.save(opt.vname,validation)
