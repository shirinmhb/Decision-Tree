import numpy as np
from copy import deepcopy
from statistics import stdev

class Node:
  def __init__(self):
    #value parent attribute
    self.value = None
    #best attribute to seperate this node
    self.attribute = None
    # self.next = None
    self.childs = None
    #only for leaf
    self.label = None
  
class ID3:
  def __init__(self):
    self.availableAttributes = list(range(1,23))

  def cal_entropy(self, list1):
    count_e = np.count_nonzero(list1 == 'e', axis = 0)
    count_p = np.count_nonzero(list1 == 'p', axis = 0)
    pr_e = count_e / (count_e + count_p)
    pr_p = count_p / (count_e + count_p)
    if pr_p == 0 or pr_e == 0:
      return 0
    ent = (-1 * pr_e * np.log2(pr_e)) + (-1 * pr_p * np.log2(pr_p))
    return ent
  
  def cal_gain(self, attributeIndex, domain, rootData):
    listNode = []
    for k in range(len(domain)):
      listNode.append([])
    for i in range(len(rootData)):
      for j in range(len(domain)):
        if rootData[i][attributeIndex] == domain[j]:
          listNode[j].append(rootData[i])
          break
    listNode = np.array(listNode)
    p = self.cal_entropy(np.array(rootData)[:,0])
    splitinfo = 0
    for i in range(len(listNode)):
      ni = (len(listNode[i])/len(rootData))
      c = ni * self.cal_entropy(np.array(listNode[i])[:,0])
      # print(c, (len(listNode[i])/len(rootData)), self.cal_entropy(np.array(listNode[i])[:,0]))
      p -= c
      # splitinfo -= (ni * np.log(ni))
    return (p,listNode)

  def select_attribute(self, root):
    gain = []
    lists = []
    uniqValues = []
    maxGain = -1
    indexMaxGain = -1
    index = 0
    atrIndex = 0
    for i in self.availableAttributes:
      u, c = np.unique(np.array(root)[:,i], return_counts=True)
      g, seperatedList = self.cal_gain(i, u, root)
      gain.append(g)
      lists.append(seperatedList)
      uniqValues.append(u)
      if g >= maxGain:
        maxGain = g
        indexMaxGain = index
        atrIndex = i
      index += 1
    self.availableAttributes.remove(atrIndex)
    return (atrIndex, lists[indexMaxGain], uniqValues[indexMaxGain])

  def make_tree(self, node, data):
    u, c = np.unique(np.array(data)[:,0], return_counts=True)
    if len(u) == 1: #its pure
      node.label = u[0]
      return node
    attrib, listChilds, domainAttr = self.select_attribute(data)
    node.attribute = attrib
    node.childs = []
    for i in range(len(domainAttr)):
      child = Node()
      child.value = domainAttr[i]
      node.childs.append(self.make_tree(child, listChilds[i]))
    return node
  
  def print_tree(self, node, i):
    print("level:", i, ", value:", node.value, ", attr:", node.attribute, ", label:", node.label)
    if node.childs != None:
      for ch in node.childs:
        self.print_tree(ch, i+1)

  def classifier(self, sample, root):
    if root.label != None:
      return root.label
    a = sample[root.attribute]
    for child in root.childs:
      if child.value == a:
        return (self.classifier(sample, child))

def read_data():
  with open('Agaricus-lepiota.data.txt') as f:
    content = [line.strip().split(',') for line in f] 
  content = np.array(content)
  return(retrive_missing(content))

def retrive_missing(content):
  for i in range(1, len(content[0])):
    u, c = np.unique(content[:,i], return_counts=True)
    if '?' in u:
      modeIndex = np.argmax(c)
      mode = u[modeIndex]
      if mode == '?':
        u = np.delete(u, modeIndex)
        c = np.delete(c, modeIndex)
        modeIndex = np.argmax()
        mode = u[modeIndex]
    for j in range(len(content)):
      if content[j][i] == '?':
        content[j][i] = mode
  return content
      
def v10fold(data):
  data = data.tolist()
  accuracies = []
  step = int(len(data)/10)
  for i in range(0, 10):
    testData = np.array(data[i*step:(i+1)*step])
    trainData = np.array(data[:i*step]+data[(i+1)*step:])
    accuracies.append(cal_accuracy(trainData, testData))
  mean = sum(accuracies)/len(accuracies)
  std = stdev(accuracies)
  print(mean, "+/-", std)

def cal_accuracy(train, test):
  root = Node()
  root.value = "root"
  id3 = ID3()
  tree = id3.make_tree(root, train)
  correct = 0
  for sample in test:
    label = id3.classifier(sample, tree)
    if sample[0] == label:
      correct += 1
  return (correct*100/len(test))

data = read_data()
root = Node()
root.value = "root"
id3 = ID3()
tree = id3.make_tree(root, data)
id3.print_tree(root, 0)
# v10fold(data)
