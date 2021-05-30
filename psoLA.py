import numpy as np
import pandas as pd
import random
import math,time,sys,os
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
from copy import deepcopy
from sklearn.cluster import KMeans
#========================================================================================================================

def sigmoid1(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))


def Vfunction1(gamma):
	return abs(np.tanh(gamma))

def Vfunction3(gamma):
	val = 1 + gamma*gamma
	val = math.sqrt(val)
	val = gamma/val
	return abs(val)




def fitness(position,trainX,trainy,testX,testy):
	cols=np.flatnonzero(position)

	train_label = trainy[cols]
	unique, counts = np.unique(train_label, return_counts=True)
	# print("train label:",dict(zip(unique, counts)))
	if len(counts) != 2:
		print("Error in class no")
		return 100
	x = counts[0]
	y = counts[1]
	if x>y:
		IR = x/y
	else:
		IR = y/x

	val=100
	if np.shape(cols)[0]==0:
		return val	
	# clf=KNeighborsClassifier(n_neighbors=5)
	clf = MLPClassifier(hidden_layer_sizes=np.shape(trainX)[1],max_iter=2000) 
	#clf = RandomForestClassifier(n_estimators=100)
	train_data=trainX[cols,:]
	test_data=testX.copy()
	train_label = trainy[cols]
	test_label = testy.copy()
	clf.fit(train_data,train_label)
	# val=1-clf.score(test_data,test_label)
	pred_label=clf.predict(test_data)
	f1Score = 1-f1_score(test_label,pred_label,average='macro')

	# unique, counts = np.unique(test_label, return_counts=True)
	# print("test label:",dict(zip(unique, counts)))

	pred_prob = clf.predict_proba(test_data)
	aucScore = 1 - roc_auc_score(test_label ,pred_prob[:,1])

	val = 0.45 * f1Score + 0.45*aucScore + 0.1*IR #+ 0.01*(sum(position)/len(trainy))
	# print("fitval:",val)
	return val

def onecount(position):
	cnt=0
	for i in position:
		if i==1.0:
			cnt+=1
	return cnt


def allfit(population,trainX,trainy,testX,testy):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],trainX,trainy,testX,testy)     
		#print(acc[i])
	return acc

def initialize(popSize,dim,trainX,trainy):
	population=np.zeros((popSize,dim))
	minn = 1
	maxx = math.floor(0.1*dim)
	if maxx<minn:
		minn = maxx
	
	for i in range(popSize):
		random.seed(i**3 + 19 + 83*time.time() ) 
		no = random.randint(minn,maxx)
		if no == 0:
			no = 1
		random.seed(time.time()*37 + 29)
		pos = random.sample(range(0,dim-1),no)
		for j in pos:
			population[i][j]=1
		
		# print(population[i])  
		
	return population

def initializeVelocity(popSize,dim):
	population=np.zeros((popSize,dim))
	for i in range(popSize):
		for j in range(dim):
			random.seed(i**3 + time.time() ) 
			population[i][j]=random.uniform(-1,1)				
	return population

def initializeguided(popSize,dim,trainX,trainy):
	unique, counts = np.unique(trainy, return_counts=True)
	# print("train label:",dict(zip(unique, counts)))
	if len(counts) != 2:
		print("Error in class no")
		return 100
	minoritycount = -1
	x = counts[0]
	y = counts[1]
	if x>y:
		IR = x/y
		minoritylabel = unique[1]
		minoritycount = y
	else:
		IR = y/x
		minoritylabel = unique[0]
		minoritycount = x
	print("Imbalance Ratio:",IR)
	if IR<1.5:
		#we can eliminate minority samples
		return initialize(popSize,dim)
	#else:
	population=np.zeros((popSize,dim))
	for i in range(popSize):
		random.seed(time.time()*37 + 29)
		pos = random.sample(range(0,np.shape(trainX)[0]- minoritycount - 1),minoritycount)
		index = 0
		for j in range(dim):
			if trainy[j] == minoritylabel:
				population[i][j] = 1
			else:
				if index in pos:
					population[i][j]=1
				index += 1
		
	return population


def clusteringInit(trainX,trainy):
	dimension = np.shape(trainX)[0]
	unique, counts = np.unique(trainy, return_counts=True)
	# print("train label:",dict(zip(unique, counts)))
	if len(counts) != 2:
		print("Error in class no")
		return 100
	minoritycount = -1
	majoritylabel = -1
	minoritylabel = -1
	x = counts[0]
	y = counts[1]
	if x>y:
		IR = x/y
		minoritylabel = unique[1]
		majoritylabel = unique[0]
		minoritycount = y
	else:
		IR = y/x
		minoritylabel = unique[0]
		majoritylabel = unique[1]
		minoritycount = x
	# print("Imbalance Ratio:",IR)
	alpha = 2
	if IR<2:
		alpha = 1
	clustercount = min(int(minoritycount/2),10)
	kmeans = KMeans(n_clusters=clustercount).fit(trainX)
	# print(kmeans.labels_)
	clusterLabels = deepcopy(kmeans.labels_)
	 

	Ratiosum = 0
	MAsum = 0
	MIsum = 0
	for i in range(clustercount):
		MAi = 0
		MIi = 0
		for index in range(dimension):
			if clusterLabels[index] == i:
				if trainy[index] == majoritylabel:
					MAi += 1
				elif trainy[index] == minoritylabel:
					MIi += 1
		if MIi == 0:
			MIi = 1
		MAsum += MAi
		MIsum += MIi

		Ratiosum += (MAi/MIi)

	MAtrainindices = []
	for i in range(clustercount):
		MAi = 0
		MIi = 0
		iCluster = []
		for index in range(dimension):
			if clusterLabels[index] == i:
				if trainy[index] == majoritylabel:
					iCluster.append(index)
					MAi += 1
				elif trainy[index] == minoritylabel:
					MIi += 1
		if MIi == 0:
			MIi = 1
		# SSizei = (alpha*MIi)*(MAi)/(MIi*Ratiosum)
		SSizei = (alpha*MIi)*(MAi*MIsum)/(MIi*MAsum)
		takei = random.sample(iCluster,math.ceil(SSizei))
		MAtrainindices = deepcopy( MAtrainindices + takei )
	# print(np.shape(MAtrainindices))

	MItrainindices = []	
	for index in range(dimension):
		if trainy[index] == minoritylabel:
			MItrainindices.append(index)
	# print(np.shape(MItrainindices))

	trainindices = deepcopy(MAtrainindices+ MItrainindices )
	return trainindices

def initializeClustering(popSize,dim,trainX,trainy):
	population=np.zeros((popSize,dim))
	for inx in range(popSize):
		trainindices = clusteringInit(trainX,trainy)
		for i in trainindices:
			population[inx][i] = 1
	return population


def randomwalk(agent):
	percent = 30
	percent /= 100
	neighbor = agent.copy()
	size = np.shape(agent)[0]
	upper = int(percent*size)
	if upper <= 1:
		upper = size
	x = random.randint(1,upper)
	pos = random.sample(range(0,size - 1),x)
	for i in pos:
		neighbor[i] = 1 - neighbor[i]
	return neighbor

def adaptiveBeta(agent,agentFit, trainX, trainy,testX,testy):
	bmin = 0.1 #parameter: (can be made 0.01)
	bmax = 1
	maxIter = 10 # parameter: (can be increased )
	
	for curr in range(maxIter):
		neighbor = agent.copy()
		size = np.shape(agent)[0]
		neighbor = randomwalk(neighbor)

		beta = bmin + (curr / maxIter)*(bmax - bmin)
		for i in range(size):
			random.seed( time.time() + i )
			if random.random() <= beta:
				neighbor[i] = agent[i]
		neighFit = fitness(neighbor,trainX,trainy,testX,testy)
		if neighFit <= agentFit:
			agent = neighbor.copy()
			agentFit = neighFit
			
	return (agent,agentFit)

def mutation(agent,agentFit, trainX, trainy,testX,testy):
	mutationrate = 0.1
	neighbor = agent.copy()
	size = np.shape(agent)[0]
	upper = int(mutationrate*size)
	if upper <= 1:
		upper = size
	random.seed(time.time())
	x = random.randint(1,upper)
	pos = random.sample(range(0,size - 1),x)
	for i in pos:
		neighbor[i] = 1 - neighbor[i]
	neighFit = fitness(neighbor,trainX,trainy,testX,testy)

	if neighFit>agentFit:
		neighbor=agent.copy()
		neighFit=agentFit
	return neighbor,neighFit

def toBinary(currAgent,currVelo,prevAgent,dimension,trainX,trainy,testX,testy):
	# print("continuous",solution)
	# print(prevAgent)
	
	Xnew = np.zeros(np.shape(currAgent))
	for i in range(dimension):
		random.seed(time.time()+i)
		temp = sigmoid1(currAgent[i])
		
		# if temp > 0.5: # sfunction
		# 	Xnew[i] = float(1)
		# else:
		# 	Xnew[i] = float(0)
		temp = Vfunction3(currVelo[i])		
		if temp > 0.5: # vfunction
			Xnew[i] = 1 - prevAgent[i]
		else:
			Xnew[i] = prevAgent[i]
	return Xnew

def createReduced(datasetname,trainX,trainy,testX,testy,answer):
	print(datasetname)
	cols = np.flatnonzero(answer)
	train_data=trainX[cols,:]
	test_data=testX.copy()
	train_label = trainy[cols]
	test_label = testy.copy()

	#store
	#Header
	#Data:Label
	(a,b)=np.shape(train_data)
	print("new train shape",a,b)
	newname = "train"+datasetname+".csv"
	with open(newname,"w") as f:
		for i in range(b):
			print("attr"+str(i+1),end=',',file=f)
		print("class",file=f)
		for row in range(np.shape(train_data)[0]):
			for y in train_data[row]:
				print(y,end=',',file=f)
			print(str(int(train_label[row])),file=f)
	
	(a,b)=np.shape(test_data)
	print("new test shape",a,b)
	newname = "test"+datasetname+".csv"
	with open(newname,"w") as f:
		for i in range(b):
			print("attr"+str(i+1),end=',',file=f)
		print("class",file=f)
		for row in range(np.shape(test_data)[0]):
			for y in test_data[row]:
				print(y,end=',',file=f)
			print(str(int(test_label[row])),file=f)


def updateLA(prevDec,beta,pvec):
	a= 0.01
	b= 0.01
	r=3 # 3 decisions
	if beta==0: # reward
		for j in range(3): # 3 decisions
			if j-1 == prevDec:
				pvec[j]=pvec[j]+a*(1-pvec[j])
			else:
				pvec[j]=(1-a)*pvec[j]
	elif beta==1: #penalty
		for j in range(3): # 3 decisions
			if j-1 == prevDec:
				pvec[j]=(1-b)*pvec[j]
			else:
				pvec[j]= b/(r-1)+ (1-b)*pvec[j]
	# decision = np.random.choice([1,-1,0],1,p=pvec)
	return pvec

#==============================================================================================================
def funcPSO(popSize,maxIter,filename,C1,C2,W):
	df=pd.read_csv(filename)
	(a,b)=np.shape(df)
	print(a,b)
	data = df.values[:,0:b-1]
	label = df.values[:,b-1]
	#scaling
	scaler = MinMaxScaler()
	scaler.fit(data)
	data = scaler.transform(data)
	# dimension = np.shape(data)[0] #particle dimension

	cross = 5
	#if a>900:
	#	cross = 10
	test_size = (1/cross)
	trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size)
	print(np.shape(trainX),np.shape(testX),np.shape(trainy),np.shape(testy))
	# print(testy)
	# arr1inds = trainy.argsort()
	# trainy = trainy[arr1inds]
	# trainX = trainX[arr1inds]
	# clf=KNeighborsClassifier(n_neighbors=5)
	# clf = RandomForestClassifier(n_estimators=100)
    clf = MLPClassifier(hidden_layer_sizes=np.shape(trainX)[1],max_iter=2000)
	clf.fit(trainX,trainy)
	val=clf.score(testX,testy)
	whole_accuracy = val
	print("Total Acc: ",val)
	dimension = np.shape(trainX)[0]

	x_axis = []
	y_axis = []

	population = initialize(popSize,dimension,trainX,trainy) 
	velocity = initializeVelocity(popSize,dimension)
	LAC1 = np.zeros((popSize,3))
	LAC2 = np.zeros((popSize,3))
	LAW = np.zeros((popSize,3))
	for i in range(popSize):
		LAC1[i][0] = (1/3)
		LAC1[i][1] = (1/3)
		LAC1[i][2] = (1/3)

		LAC2[i][0] = (1/3)
		LAC2[i][1] = (1/3)
		LAC2[i][2] = (1/3)

		LAW[i][0] = (1/3)
		LAW[i][1] = (1/3)
		LAW[i][2] = (1/3)
	# print(population)
	gbestVal = 1000
	gbestVec = np.zeros(np.shape(population[0])[0])

	pbestVal = np.zeros(popSize)
	pbestVec = np.zeros(np.shape(population))	
	for i in range(popSize):
		pbestVal[i] = 1000
	
	start_time = datetime.now()
	fitList = allfit(population,trainX,trainy,testX,testy)

	for curIter in range(maxIter):
		popnew = np.zeros((popSize,dimension))
		fitListnew =[]
		# fitList = allfit(population,trainX,trainy,testX,testy)
		# for i in range(popSize):
			# print('before:',population[i].sum(),fitList[i])
			# population[i],fitList[i] = adaptiveBeta(population[i],fitList[i],trainX,trainy,testX,testy)
			# population[i],fitList[i] = deepcopy(mutation(population[i],fitList[i],trainX,trainy,testX,testy))
			# print('after:',population[i].sum(),fitList[i])
		
		#update pbest
		for i in range(popSize):
			if (fitList[i] < pbestVal[i]):
				pbestVal[i] = fitList[i]
				pbestVec[i] = population[i].copy()
				# print("pbest updated")

		#update gbest
		for i in range(popSize):
			# print('%.4f'%fitList[i],population[i].sum())
			if (fitList[i] < gbestVal):
				gbestVal = fitList[i]
				gbestVec = population[i].copy()
		# print('')
		# print(gbestVec)
		print(curIter,"gbest: ",gbestVal,gbestVec.sum())
		#update W
		# W = WMAX - (curIter/maxIter)*(WMAX - WMIN )
		# print("w: ",W)
		for inx in range(popSize):
			#inx <- particle index
			random.seed(time.time()+13)
			decisionW = np.random.choice([-1,0,1],1,p=LAW[inx])[0]
			W = W + decisionW*deltaW
			if W>1:
				W=1
			if W<0:
				W=0

			random.seed(time.time()+17)
			decisionC1 = np.random.choice([-1,0,1],1,p=LAC1[inx])[0]
			C1 = C1 + decisionC1*deltaC
			if C1>10:
				C1=10
			if C1<0:
				C1=0
			r1 = C1 * random.random()

			random.seed(time.time()+19)
			decisionC2 = np.random.choice([-1,0,1],1,p=LAC2[inx])[0]
			C2 = C2 + decisionC2*deltaC
			if C2>10:
				C2=10
			if C2<0:
				C2=0
			r2 = C2 * random.random()
			# print(W,C1,C2)

			x = np.subtract(pbestVec[inx] , population[inx])
			y = np.subtract(gbestVec , population[inx])
			velocity[inx] = np.multiply(W,velocity[inx]) + np.multiply(r1,x) + np.multiply(r2,y)
			popnew[inx] = population[inx]+velocity[inx]
			popnew[inx] = toBinary(popnew[inx],velocity[inx],population[inx],dimension,trainX,trainy,testX,testy)

			fitNew = fitness(popnew[inx],trainX,trainy,testX,testy)    
			fitListnew.append(fitNew)

			beta=1 #penalty
			if fitNew<fitList[inx]:#reward 
				beta = 0
			LAC1[inx]= deepcopy(updateLA(decisionC1,beta,LAC1[inx]))
			LAC2[inx]= deepcopy(updateLA(decisionC2,beta,LAC2[inx]))
			LAW[inx]= deepcopy(updateLA(decisionW,beta,LAW[inx]))
			

		fitList= deepcopy(fitListnew)

		population = popnew.copy()

	time_required = datetime.now() - start_time
	output = gbestVec.copy()
	# print(output)


	cols=np.flatnonzero(output)
	#print(cols)
	train_data=trainX[cols,:]
	test_data=testX.copy()
	train_label = trainy[cols]
	# clf = RandomForestClassifier(n_estimators=300)
	clf = MLPClassifier(hidden_layer_sizes=np.shape(trainX)[1],max_iter=5000) 
	clf.fit(train_data,train_label)
	pred_label=clf.predict(test_data)
	f1Score = f1_score(testy,pred_label,average='macro')

	pred_prob = clf.predict_proba(test_data)
	aucScore = roc_auc_score(testy,pred_prob[:,1])

	val = f1Score + aucScore
	print(val,onecount(output))

	return val,aucScore,f1Score,output,trainX,trainy,testX,testy


#========================================================================================================================
omega = 0.9
popSize = 20
maxIter = 100
C1 = 2
C2 = 2
WMAX=0.9
WMIN=0.4
W = 0.9
deltaC = 0.4
deltaW = 0.1


warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

directory="datasets/" #"
# filelist= sorted(os.listdir(directory )) "Wine.csv"
filelist = ["pima","segment0","led7digit","abalone9-18","poker-9_vs_7","yeast5","abalone19","page-blocks0"]
# filelist = ["abalone9-18"]
print(filelist)

for filename in filelist:
	print(filename)
	best_accuracy = -1
	bestAuc = -1
	bestF1s = -1
	best_no_features = -1
	average_accuracy = 0
	global_count = 0
	accuracy_list = []
	aucscoreList = []
	f1scoreList = []

	for global_count in range(20):
		print("run no:",global_count)
		val,valAuc,valF1,output,trnX,trny, tstX,tsty = funcPSO(popSize,maxIter,directory+filename+".csv",C1,C2,W)

		accuracy_list.append(val)
		aucscoreList.append(valAuc)
		f1scoreList.append(valF1)
		if ( val == best_accuracy ) and ( output.sum() < best_no_features ):
			best_accuracy = val
			bestAuc = valAuc
			bestF1s = valF1
			best_no_features = output.sum()
			# createReduced(filename,trnX,trny,tstX,tsty,output)
			# best_time_req = time_required
			# best_whole_accuracy = whole_accuracy

		if val > best_accuracy :
			best_accuracy = val
			bestAuc = valAuc
			bestF1s = valF1
			best_no_features = output.sum()
			# createReduced(filename,trnX,trny,tstX,tsty,output)
			# best_time_req = time_required
			# best_whole_accuracy = whole_accuracy

	print('best: ',best_accuracy, best_no_features)
	accuracy_list=np.array(accuracy_list)
	aucscoreList=np.array(aucscoreList)
	f1scoreList=np.array(f1scoreList)

	arr1inds = np.argsort(aucscoreList)
	aucscoreList=aucscoreList[arr1inds[::-1]]
	f1scoreList=f1scoreList[arr1inds[::-1]]
	accuracy_list=accuracy_list[arr1inds[::-1]]

	print(accuracy_list)
	print(aucscoreList)
	print(f1scoreList)

	aucscoreList=aucscoreList[:5]
	f1scoreList=f1scoreList[:5]
	avgAuc = np.mean(aucscoreList)
	stdAuc = np.std(aucscoreList)

	avgF1 = np.mean(f1scoreList)
	stdF1 = np.std(f1scoreList)
	# temp=sys.argv[1].split('/')[-1]
	temp = filename.split('.')[0]
	with open("result_PSOLA.csv","a") as f:
		print(temp,"%.4f" % bestAuc,"%.4f" % avgAuc,"%.4f" % stdAuc,"%.4f" % bestF1s,"%.4f" % avgF1,"%.4f" % stdF1,file=f)
