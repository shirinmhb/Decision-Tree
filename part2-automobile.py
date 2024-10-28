import numpy as np
from copy import deepcopy
from statistics import stdev
from sklearn.metrics import mean_squared_error
import random

class Node:
  def __init__(self):
    #value parent attribute
    self.value = None
    #best attribute to seperate this node
    self.attribute = None
    # self.next = None
    self.childs = None
    #only for leaf
    self.avg = None
  
class ID3:
  def __init__(self):
    self.availableAttributes = list(range(1,12))

  def cal_std(self, list1):
    list1 = list1.astype(np.float)
    if len(list1) != 1:
      return stdev(list1)
    return 0
  
  def cal_std_gain(self, attributeIndex, domain, rootData):
    listNode = []
    for k in range(len(domain)):
      listNode.append([])
    for i in range(len(rootData)):
      for j in range(len(domain)):
        if rootData[i][attributeIndex] == domain[j]:
          listNode[j].append(rootData[i])
          break
    listNode = np.array(listNode)
    p = self.cal_std(np.array(rootData)[:,0])
    for i in range(len(listNode)):
      ni = (len(listNode[i])/len(rootData))
      c = ni * self.cal_std(np.array(listNode[i])[:,0])
      p -= c
    return (p, listNode)

  def select_attribute(self, root):
    gain = []
    lists = []
    uniqValues = []
    maxGain = -1
    indexMaxGain = -1
    index = 0
    atrIndex = self.availableAttributes[0]
    for i in self.availableAttributes:
      u, c = np.unique(np.array(root)[:,i], return_counts=True)
      g, seperatedList = self.cal_std_gain(i, u, root)
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

  def cal_termination(self, list1):
    list1 = list1.astype(np.float)
    avg = np.average(list1)
    if len(list1) == 1:
      return 0
    std = stdev(list1)
    return(std/avg)*100

  def make_tree(self, node, data):
    np.unique(np.array(data)[:,0], return_counts=True)
    termination = self.cal_termination(np.array(data)[:,0])
    if termination <= 23 or len(data) <= 8 : #its pure
      list1 = np.array(data)[:,0]
      list1 = list1.astype(np.float)
      node.avg = np.average(list1)
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
    print("level:", i, ", value:", node.value, ", attr:", node.attribute, ", label:", node.avg)

    if node.childs != None:
      for ch in node.childs:
        self.print_tree(ch, i+1)

  def classifier(self, sample, root):
    if root.avg != None:
      return root.avg
    a = sample[root.attribute]
    s = []
    for child in root.childs:
      # if child.avg != None:
      #   s.append(child.avg)
      if child.value == a:
        return (self.classifier(sample, child))


def read_data():
  with open('Automobile2.txt') as f:
    content = [line.strip().split('\t') for line in f] 
  for i in range(len(content)):
    # print(content[i][len(content[i])-1:] ,content[i][:len(content[i])-1])
    content[i] = content[i][len(content[i])-1:] + content[i][:len(content[i])-1]
    content[i][0] = float(content[i][0])
  random.shuffle(content)
  content = np.array(content)
  return(content)
    
def cal_mse():
  data = read_data()
  node = Node()
  node.value = "root"
  idTree = ID3()
  train = data[:140]
  test = data[140:]
  tree = idTree.make_tree(node, train)
  idTree.print_tree(tree, 0)
  mse = 0
  y_true = []
  y_pred = []
  for sample in test:
    pred = idTree.classifier(sample, tree)
    if pred != None:
      y_pred.append(pred)
      y_true.append(sample[0].astype(np.float))
  # print(y_true, y_pred)
  return mean_squared_error(y_true, y_pred)

cal_mse()
results = []
for i in range(10):
  results.append(cal_mse())

average = sum(results)/len(results)
average = "{:e}".format(average)
std = stdev(results)
std = "{:e}".format(std)
print("mse:", average, "+/-", std)
