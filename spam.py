import glob


path =r'C:\Users\subrafive\SpamPro\TRAINING_BODY' # use your path
allFiles = glob.glob(path + "/*.eml")

test_path =r'C:\Users\subrafive\SpamPro\TESTING_BODY' # use your path
allTestFiles = glob.glob(path + "/*.eml")



#Read email labels
def read_email_labels():
  email_labels = []
  with open('C:\\Users\\subrafive\\SpamPro\\SPAMTrain.label') as sf:
      lines = sf.readlines()
      
  
  for x in lines:
    email_labels.append(x.split()[0])

    
  return email_labels

  
#Create dictionary of training data

def create_train_email_list():

  train_email_list = []
  cnt = 0
  email_labels = read_email_labels()

  for file_ in allFiles:
    with open(file_,encoding="Latin1") as f:
      lines = f.read()
      word_list = lines.split()
      d=dict()
      d['id'] = cnt
      d['body'] = word_list
      d['label'] = email_labels[cnt]
      train_email_list.append(d)
      cnt = cnt + 1
  numTrain = int(round(0.7*len(train_email_list)))
  numTest = int(round(0.3*len(train_email_list)))
  trainData = train_email_list[0:numTrain]
  testData = train_email_list[numTrain:numTest+numTrain]
  print('No.of train',numTrain)
  print('No.of test',numTest)
  
  return trainData,testData
  
  
def create_test_email_list():
  
  test_email_list = []
  cnt = 0
  
  for file_ in allTestFiles:
    with open(file_,encoding="Latin1") as f:
      lines = f.read()
      word_list = lines.split()
      d=dict()
      d['id'] = cnt
      d['body'] = word_list
      test_email_list.append(d)
      cnt = cnt + 1
  return test_email_list
  

def train(trainData):
    global pA
    global pNotA
    total = 0
    numSpam = 0
    for email in trainData:
        if email['label'] == '0':
            numSpam += 1
        total += 1
        processEmail(email['body'], email['label'])
    
    print('Number of spams:',numSpam)
    print('Total:',total)
    pA = numSpam/total
    pNotA = (total - numSpam)/total
  
  
def processEmail(body, label):
    global trainPositive
    global trainNegative
    global positiveTotal
    global negativeTotal
    
    for word in body:
        if label == '0':
            trainPositive[word] = trainPositive.get(word, 0) + 1
            positiveTotal = positiveTotal + 1
        else:
            trainNegative[word] = trainNegative.get(word, 0) + 1
            negativeTotal = negativeTotal + 1  
    
    
def conditionalWord(word, spam):
    global numWords
    alpha = 1
    if spam:
      return (trainPositive.get(word,0)+alpha)/(positiveTotal+alpha*numWords)
    return (trainNegative.get(word,0)+alpha)/(negativeTotal+alpha*numWords)
    
    
def conditionalEmail(body, spam):
    result = 1.0
    for word in body:
        result *= conditionalWord(word, spam)
    return result
    
    
def classify(email):
    isSpam = pA * conditionalEmail(email, True) # P (A | B)
    notSpam = pNotA * conditionalEmail(email, False) # P(¬A | B)
    return isSpam > notSpam

    
trainData,testData = create_train_email_list()
#testData = create_test_email_list()
pA = 0
pNotA = 0
trainPositive = {}
trainNegative = {}
positiveTotal = 0
negativeTotal = 0

#Get the number of distinct words in 
numWords = len(trainPositive)
for word in trainNegative:
  if word not in trainPositive:
    numWords = numWords + 1


#Call training function
train(trainData)
print(len(trainData),len(testData))

predictions = []
#Classify testing set
with open('Output.txt','w') as f:
  for email in testData:
    pred = classify(email['body'])
    if(pred):
      predictions.append("0")
    else:
      predictions.append("1")
    f.write('%s : %s\n' %(email['id'],pred))
    
#Calculate Accuracy
cnt = 0
correct = 0
for email in testData:
  if email['label'] == predictions[cnt]:
    correct = correct + 1

accuracy = (correct/len(testData))*100
print('Accuracy :',accuracy)    
print(type(predictions[0]),type(testData[0]['label']))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#Read up the wikipedia article on spam filtering with naive bayes 
#Do train-test split to evaluate current model performance   
#Try improvements suggested (atleast one) evaluate if performance has improved
#Create ui Chrome extension
  
  
