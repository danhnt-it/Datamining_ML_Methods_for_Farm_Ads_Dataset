#import necessary libraries
import string
# import os
# print(os.getcwd())
import re #remove elements
import pickle #save model

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import numpy as np

import tkinter as tk #create UI
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def read_dataset(link):
	"""
	Read Farm-Ads Dataset
	input: link of dataset
	output: a list of data, each line is 1 element of list named data
	"""
	data = [] 
	with open(link, 'r') as f: 
		while True:
			line = f.readline()
			if not line:
				break
			data.append(line)
	return data

def split_labels(data):
	"""
	Split labels
	input: dataset
	output: labels of all data lines, 0 and 1 format
	"""
	labels_ori = []
	for line in data:
		new = line.split()[0]
		labels_ori.append(int(new))
	# replace -1 with 0 for easy later handling
	labels = []
	for i in labels_ori:
		if i == -1:
			labels.append(0)
		else: labels.append(i)
	return labels

def clear_prefix(data):
	"""
	Clear labels and prefix ad- title- header- from data
	input: dataset
	output: data without prefix and labels
	"""
	data_cleared_prefix = []
	for line in data:
		line = re.sub(line.split()[0], '', line)
		line = re.sub('ad-', '', line)
		line = re.sub('title-', '', line)
		line = re.sub('header-', '', line)
		data_cleared_prefix.append(line)
	return data_cleared_prefix

def clear_singular_test_prefix(data):
	"""
	Clear prefix ad- title- header- from test data
	input: test data
	output: data without prefix
	"""
	data = re.sub('ad-', '', data)
	data = re.sub('title-', '', data)
	data = re.sub('header-', '', data)
	return data

def clear_multiple_test_prefix_without_labels(data):
	"""
	Clear labels and prefix ad- title- header- from data
	input: dataset
	output: data without prefix and labels
	"""
	data_cleared_prefix = []
	for line in data:
		line = re.sub('ad-', '', line)
		line = re.sub('title-', '', line)
		line = re.sub('header-', '', line)
		data_cleared_prefix.append(line)
	return data_cleared_prefix

"""
EVALUATION INIT
"""
def find_TP(y_pred, y_test):
	"""
	find true positive TP: predict positive + labels positive
	input: y predict, y label
	output: number of TP points
	"""
	count = 0
	Positive = np.multiply(y_pred, y_test)
	return np.count_nonzero(Positive)

def find_P(y_test):
	"""
	find Positive point: labels positive
	input: y label
	output: number of y label positive
	"""
	return np.count_nonzero(y_test)

def find_FP(y_pred, y_test):
	"""
	find false positive TP: predict positive + labels negative
	input: y predict, y label
	output: number of FP points
	"""
	count = 0
	for i in range(y_pred.shape[0]):
			if y_pred[i] == 1:
					if y_test[i] == 0:
							count += 1
	return count

def Recall(TP, P):
	"""
	Recall: TP/TP+FN, from all model predicted is positive, how many of them is positive among the truly positive
	input: TP, P
	output: recall
	"""
	return 1.0*TP/P

def Precision(TP, FP):
	"""
	Precision: TP/TP+FP, from all model predicted is positive, how many of them is truly positive
	input: TP, FP
	output: precision
	"""	
	return 1.0*TP/(TP+FP) 

def F1_score(Precision, Recall):
	"""
	combination of Precision and Recall
	input: precision, recall
	output: F1score as 2precision*recall/(precision+recall)
	"""
	return (Precision*Recall)/(Precision+Recall)*2

def Accuracy(y_pred, y_test): 
	"""
	Accuracy: how many precent of the prediction that model made is as the result labels
	input: y predict, y labels
	output: percentage (0 to 1)
	"""
	count = 0
	for i in range(y_pred.shape[0]):
			if y_pred[i] == y_test[i]:
					count += 1
	return count/y_pred.shape[0]


"""
CLASSIFICATION
"""
def tfidf(data_cleared_prefix, labels, position_to_split_dataset, active_folder_link):
	"""
	Ifidf Vectorizer use library function
	input: data cleared prefix and labels
	split ratio: train/test = 7/3 for example
	output: train_X, train_y, test_X, test_y
	"""
	text_transformed = TfidfVectorizer()

	train_X = text_transformed.fit_transform(data_cleared_prefix[:position_to_split_dataset])
	train_y = labels[:position_to_split_dataset]
	test_X = text_transformed.transform(data_cleared_prefix[position_to_split_dataset:])
	test_y = labels[position_to_split_dataset:]

	#save for testing use
	f = open(active_folder_link + '/tfidf.pickle', 'wb')
	pickle.dump(text_transformed, f)
	f.close()

	return train_X, train_y, test_X,test_y


def Logistic_Regression(train_X, train_y, active_folder_link):
	"""
	Logistic Regression
	"""
	global predict_logist
	logist = LogisticRegression()
	logist.fit(train_X, train_y)
	predict_logist = logist.predict(test_X)

	# save model
	f = open(active_folder_link + '/logistic_classifier.pickle', 'wb')
	pickle.dump(logist, f)
	f.close()

	return predict_logist

def Evaluate_Logistic_Regression(predict_logist, test_y):
	"""
	Evaluate Logistic Regression
	"""
	global Accu_logist, Precision_logist, Recall_logist, F1_S_logist
	print("Evaluation of Logistic Regression")
	TP_logist = find_TP(predict_logist, test_y)
	FP_logist = find_FP(predict_logist, test_y)
	P_logist = find_P(test_y)
	Accu_logist = Accuracy(predict_logist, test_y)

	Precision_logist = TP_logist/(TP_logist+FP_logist)
	Recall_logist = TP_logist/P_logist
	F1_S_logist = F1_score(Precision_logist, Recall_logist)

	print("TP = ", TP_logist)
	print("FP = ", FP_logist)
	print("P  = ", P_logist)
	print("Accuracy = ", Accu_logist)
	print("Precision = ", Precision_logist)
	print("Recall = ", Recall_logist)
	print("F1_score:", F1_S_logist)

	return Accu_logist, Precision_logist, Recall_logist, F1_S_logist


def Naive_Bayes(train_X, train_y, active_folder_link):
	"""
	Naive Bayes
	"""
	global predict_NB
	NB = BernoulliNB()
	NB.fit(train_X, train_y)
	predict_NB = NB.predict(test_X)

	#save model
	f = open(active_folder_link + '/naive_bayes_classifier.pickle', 'wb')
	pickle.dump(NB, f)
	f.close()

	return predict_NB

def Evaluate_Naive_Bayes(predict_NB, test_y):
	"""
	Evaluate Naive Bayes
	"""
	global Accu_NB, Precision_NB, Recall_NB, F1_S_NB
	print("Evaluation of Naive Bayes")
	TP_NB = find_TP(predict_NB, test_y)
	FP_NB = find_FP(predict_NB, test_y)
	P = find_P(test_y)
	Accu_NB = Accuracy(predict_NB, test_y)

	Precision_NB = TP_NB/(TP_NB+FP_NB)
	Recall_NB = TP_NB/P
	F1_S_NB = F1_score(Precision_NB, Recall_NB)

	print("TP = ", TP_NB)
	print("FP = ", FP_NB)
	print("P  = ", P)
	print("Accuracy = ", Accu_NB)
	print("Precision = ", Precision_NB)
	print("Recall = ", Recall_NB)
	print("F1_score:", F1_S_NB)
	return Accu_NB, Precision_NB, Recall_NB, F1_S_NB


def Decision_Tree(train_X, train_y, active_folder_link):
	"""
	Decision Tree
	"""
	global predict_DT
	DT = DecisionTreeClassifier()
	DT.fit(train_X, train_y)
	predict_DT = DT.predict(test_X)

	#save model
	f = open(active_folder_link + '/decision_tree_classifier.pickle', 'wb')
	pickle.dump(DT, f)
	f.close()

	return predict_DT


def Evaluate_Decision_Tree(predict_DT, test_y):
	"""
	Evaluate Decision Tree
	"""
	global Accu_DT, Precision_DT, Recall_DT, F1_S_DT
	print("Evaluation of Decision Tree")
	TP_DT = find_TP(predict_DT, test_y)
	FP_DT = find_FP(predict_DT, test_y)
	P = find_P(test_y)
	Accu_DT = Accuracy(predict_DT, test_y)

	Precision_DT = TP_DT/(TP_DT+FP_DT)
	Recall_DT = TP_DT/P
	F1_S_DT = F1_score(Precision_DT, Recall_DT)

	print("TP = ", TP_DT)
	print("FP = ", FP_DT)
	print("P  = ", P)
	print("Accuracy = ", Accu_DT)
	print("Precision = ", Precision_DT)
	print("Recall = ", Recall_DT)
	print("F1_score:", F1_S_DT)
	return Accu_DT, Precision_DT, Recall_DT, F1_S_DT

def KNN(train_X, train_y, n_neighbors, active_folder_link):
	"""
	K-Nearest Neighbors
	"""
	global predict_KNN
	KNN = KNeighborsClassifier(n_neighbors = n_neighbors)
	KNN.fit(train_X, train_y)
	predict_KNN = KNN.predict(test_X)

	#save model
	f = open(active_folder_link + '/KNN_classifier.pickle', 'wb')
	pickle.dump(KNN, f)
	f.close()

	return predict_KNN


def Evaluate_KNN(predict_KNN, test_y):
	"""
	Evaluate K-Nearest Neighbors
	"""
	global Accu_KNN, Precision_KNN, Recall_KNN, F1_S_KNN
	print("Evaluation of K Nearest Neighbors")
	TP_KNN = find_TP(predict_KNN, test_y)
	FP_KNN = find_FP(predict_KNN, test_y)
	P = find_P(test_y)
	Accu_KNN = Accuracy(predict_KNN, test_y)

	Precision_KNN = TP_KNN/(TP_KNN+FP_KNN)
	Recall_KNN = TP_KNN/P
	F1_S_KNN = F1_score(Precision_KNN, Recall_KNN)

	print("TP = ", TP_KNN)
	print("FP = ", FP_KNN)
	print("P  = ", P)
	print("Accuracy = ", Accu_KNN)
	print("Precision = ", Precision_KNN)
	print("Recall = ", Recall_KNN)
	print("F1_score:", F1_S_KNN)
	return Accu_KNN, Precision_KNN, Recall_KNN, F1_S_KNN

def main():
	root = tk.Tk()
	root.active_folder_link = "" #'F:/HK5/KTDL/Project/log' #for example
	root.title("Farm-Ads Dataset Classification v0.1")
	root.geometry("715x510")
	root.configure(background='white')
	root.filename = ""
	root.testfilename = ""

	def get_active_folder_link():
		"""
		get link of the folder that saving log
		"""
		root.active_folder_link = filedialog.askdirectory()
		print(root.active_folder_link)
		return root.active_folder_link

	btn11 = tk.Button(root, text = "Save log at:", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = get_active_folder_link)
	btn11.place(x=580, y=5, height=22, width=120)

	def close_window():
		root.destroy()

	def load_dataset_link():
		root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files","*.*"),("txt files","*.txt")))
		print(root.filename)
		lbl3.config(text=root.filename, fg = "black")

	def get_content_entry(entry): #get content from the entry box
		print(entry.widget.get())

	def take_position_to_split_dataset(labels): #compute the position to split dataset at input ratio
		temp = ratio_str.get()
		temp = temp.split("/")
		train_ratio = int(temp[0])
		test_ratio = int(temp[1])
		position_to_split = int(len(labels)*train_ratio/(train_ratio+test_ratio))
		return position_to_split

	def preprocess_data():
		global data_cleared_prefix, train_X, train_y, test_X, test_y
		global labels
		data = read_dataset(root.filename)
		labels = split_labels(data)

		position_to_split_dataset = take_position_to_split_dataset(labels)
		print("Position to split dataset: " + str(position_to_split_dataset))
		
		data_cleared_prefix = clear_prefix(data)
		train_X, train_y, test_X, test_y = tfidf(data_cleared_prefix, labels, position_to_split_dataset, root.active_folder_link)
		print("Preprocess via Tfidf completed! ")

		lbl7.config(text="Preprocess Completed!", fg = "black")
		return data_cleared_prefix, train_X, train_y, test_X, test_y


	def Logistic_Regression_button():
		predict_logist = Logistic_Regression(train_X, train_y, root.active_folder_link)
		Accu_logist, Precision_logist, Recall_logist, F1_S_logist = Evaluate_Logistic_Regression(predict_logist, test_y)
		lbl11.config(text="Precision: " + str(round(Precision_logist, 3)) + "	Recall: " + str(round(Recall_logist, 3)) + "\nF1_Score: " + str(round(F1_S_logist, 3)) + "\nAccuracy: " + str(round(Accu_logist, 3)) , fg = "black")

	def Naive_Bayes_button():
		predict_NB = Naive_Bayes(train_X, train_y, root.active_folder_link)
		Accu_NB, Precision_NB, Recall_NB, F1_S_NB = Evaluate_Naive_Bayes(predict_NB, test_y)
		lbl12.config(text="Precision: " + str(round(Precision_NB, 3)) + "	Recall: " + str(round(Recall_NB, 3)) + "\nF1_Score: " + str(round(F1_S_NB, 3)) + "\nAccuracy: " + str(round(Accu_NB, 3)) , fg = "black")

	def Decision_Tree_button():
		predict_DT = Decision_Tree(train_X, train_y, root.active_folder_link)
		Accu_DT, Precision_DT, Recall_DT, F1_S_DT = Evaluate_Decision_Tree(predict_DT, test_y)
		lbl13.config(text="Precision: " + str(round(Precision_DT, 3)) + "	Recall: " + str(round(Recall_DT, 3)) + "\nF1_Score: " + str(round(F1_S_DT, 3)) + "\nAccuracy: " + str(round(Accu_DT, 3)) , fg = "black")

	def KNN_button():
		n = n_neighbors.get()
		predict_KNN = KNN(train_X, train_y, int(n), root.active_folder_link)
		Accu_KNN, Precision_KNN, Recall_KNN, F1_S_KNN = Evaluate_KNN(predict_KNN, test_y)
		lbl14.config(text="Precision: " + str(round(Precision_KNN, 3)) + "	Recall: " + str(round(Recall_KNN, 3)) + "\nF1_Score: " + str(round(F1_S_KNN, 3)) + "\nAccuracy: " + str(round(Accu_KNN, 3)) , fg = "black")


	def load_testfile_link():
		root.testfilename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("all files","*.*"),("txt files","*.txt")))
		print(root.testfilename)

	def process_multiple_test_input():
		data_test = read_dataset(root.testfilename)
		if data_test[0].split()[0] != '1' and data_test[0].split()[0] != '-1':
			data_test_without_labels_cleared_prefix = clear_multiple_test_prefix_without_labels(data_test)

			#load save model tfidf for vectorize new data
			f1 = open(root.active_folder_link + '/tfidf.pickle', 'rb')
			text_transformed = pickle.load(f1)
			new_multiple_features = text_transformed.transform(data_test_without_labels_cleared_prefix) # new multiple features here is the new test input
			
			print("Preprocess multiple test data via Tfidf completed! ")
			print(new_multiple_features)

			model_value = value_checklist.get() #value = 1 2 3 4 Logistic Regression, Naive Bayes, Decision Tree, KNN
			print("model_value: " + str(model_value))

			if model_value is 1:
				f = open(root.active_folder_link + '/logistic_classifier.pickle', 'rb')
				logist = pickle.load(f)
				predict_test = logist.predict(new_multiple_features) # label is 0 or 1
				f.close()

				#save result file txt
				with open(root.active_folder_link + '/LR_without_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						f.write("%s\n" % item)
				
				lbl21.config(text="LR wrote a result file!", fg = "black")

			elif model_value is 2:
				f = open(root.active_folder_link + '/naive_bayes_classifier.pickle', 'rb')
				NB = pickle.load(f)
				predict_test = NB.predict(new_multiple_features) # label is 0 or 1
				f.close()

				#save result file txt
				with open(root.active_folder_link + '/NB_without_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						f.write("%s\n" % item)
				
				lbl21.config(text="NB wrote a result file!", fg = "black")

			elif model_value is 3:
				f = open(root.active_folder_link + '/decision_tree_classifier.pickle', 'rb')
				DT = pickle.load(f)
				predict_test = DT.predict(new_multiple_features) # label is 0 or 1
				f.close()

				#save result file txt
				with open(root.active_folder_link + '/DT_without_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						f.write("%s\n" % item)
				
				lbl21.config(text="DT wrote a result file!", fg = "black")

			elif model_value is 4:
				f = open(root.active_folder_link + '/KNN_classifier.pickle', 'rb')
				KNN = pickle.load(f)
				predict_test = KNN.predict(new_multiple_features) # label is 0 or 1
				f.close()

				#save result file txt
				with open(root.active_folder_link + '/KNN_without_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						f.write("%s\n" % item)
				
				lbl21.config(text="KNN wrote a result file!", fg = "black")

		else: #here is datatest with labels -1 or 1
			labels_data_test = split_labels(data_test)
			data_test_with_labels_cleared_prefix = clear_prefix(data_test)


			#load save model tfidf for vectorize new data
			f1 = open(root.active_folder_link + '/tfidf.pickle', 'rb')
			text_transformed = pickle.load(f1)
			new_multiple_features = text_transformed.transform(data_test_with_labels_cleared_prefix) # new multiple features here is the new test input
			
			print("Preprocess multiple test data via Tfidf completed! ")
			print(new_multiple_features)


			model_value = value_checklist.get() #value = 1 2 3 4 Logistic Regression, Naive Bayes, Decision Tree, KNN
			print("model_value: " + str(model_value))

			if model_value is 1:
				f = open(root.active_folder_link + '/logistic_classifier.pickle', 'rb')
				logist = pickle.load(f)
				predict_test = logist.predict(new_multiple_features) # label is 0 or 1
				f.close()

				Test_Accu_logist, Test_Precision_logist, Test_Recall_logist, Test_F1_S_logist = Evaluate_Logistic_Regression(predict_test, labels_data_test)

				#save result file txt
				with open(root.active_folder_link + '/LR_with_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						if item == 1:
							f.write("%s\n" % item)
						elif item == 0:
							f.write("%s\n" % '-1')
				
				lbl21.config(text= "LR wrote a result file!\n" + "Precision: " + str(round(Test_Precision_logist, 3)) + "	Recall: " + str(round(Test_Recall_logist, 3)) + "\nF1_Score: " + str(round(Test_F1_S_logist, 3)) + "\nAccuracy: " + str(round(Test_Accu_logist, 3)) , fg = "black")


			elif model_value is 2:
				f = open(root.active_folder_link + '/naive_bayes_classifier.pickle', 'rb')
				NB = pickle.load(f)
				predict_test = NB.predict(new_multiple_features) # label is 0 or 1
				f.close()

				Test_Accu_NB, Test_Precision_NB, Test_Recall_NB, Test_F1_S_NB = Evaluate_Logistic_Regression(predict_test, labels_data_test)

				#save result file txt
				with open(root.active_folder_link + '/NB_with_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						if item == 1:
							f.write("%s\n" % item)
						elif item == 0:
							f.write("%s\n" % '-1')
				
				lbl21.config(text= "NB wrote a result file!\n" + "Precision: " + str(round(Test_Precision_NB, 3)) + "	Recall: " + str(round(Test_Recall_NB, 3)) + "\nF1_Score: " + str(round(Test_F1_S_NB, 3)) + "\nAccuracy: " + str(round(Test_Accu_NB, 3)) , fg = "black")


			elif model_value is 3:
				f = open(root.active_folder_link + '/decision_tree_classifier.pickle', 'rb')
				DT = pickle.load(f)
				predict_test = DT.predict(new_multiple_features) # label is 0 or 1
				f.close()

				Test_Accu_DT, Test_Precision_DT, Test_Recall_DT, Test_F1_S_DT = Evaluate_Logistic_Regression(predict_test, labels_data_test)

				#save result file txt
				with open(root.active_folder_link + '/DT_with_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						if item == 1:
							f.write("%s\n" % item)
						elif item == 0:
							f.write("%s\n" % '-1')

				lbl21.config(text= "DT wrote a result file!\n" + "Precision: " + str(round(Test_Precision_DT, 3)) + "	Recall: " + str(round(Test_Recall_DT, 3)) + "\nF1_Score: " + str(round(Test_F1_S_DT, 3)) + "\nAccuracy: " + str(round(Test_Accu_DT, 3)) , fg = "black")

			elif model_value is 4:
				f = open(root.active_folder_link + '/KNN_classifier.pickle', 'rb')
				KNN = pickle.load(f)
				predict_test = KNN.predict(new_multiple_features) # label is 0 or 1
				f.close()

				Test_Accu_KNN, Test_Precision_KNN, Test_Recall_KNN, Test_F1_S_KNN = Evaluate_Logistic_Regression(predict_test, labels_data_test)

				#save result file txt
				with open(root.active_folder_link + '/KNN_with_labels_multiple_test_results.txt', 'w') as f:
					for item in predict_test:
						if item == 1:
							f.write("%s\n" % item)
						elif item == 0:
							f.write("%s\n" % '-1')
				
				lbl21.config(text= "KNN wrote a result file!\n" + "Precision: " + str(round(Test_Precision_KNN, 3)) + "	Recall: " + str(round(Test_Recall_KNN, 3)) + "\nF1_Score: " + str(round(Test_F1_S_KNN, 3)) + "\nAccuracy: " + str(round(Test_Accu_KNN, 3)) , fg = "black")



	def process_singular_test_input():
		singular_test_input = test_input.get()
		print(singular_test_input)
		singular_test_cleared_prefix = clear_singular_test_prefix(singular_test_input)
		print(singular_test_cleared_prefix)

		list_test = [] 
		list_test.append(singular_test_cleared_prefix) # because tfidf receive input as a document, not a single string

		#load save model tfidf for vectorize new data
		f1 = open(root.active_folder_link + '/tfidf.pickle', 'rb')
		text_transformed = pickle.load(f1)
		new_features = text_transformed.transform(list_test) # new features here is the new test input

		print("Preprocess test data via Tfidf completed! ")
		print(new_features)

		model_value = value_checklist.get() #value = 1 2 3 4 Logistic Regression, Naive Bayes, Decision Tree, KNN
		print("model_value: " + str(model_value))
		if model_value is 1:
			f = open(root.active_folder_link + '/logistic_classifier.pickle', 'rb')
			logist = pickle.load(f)
			predict_test = logist.predict(new_features) # label is 0 or 1
			f.close()
			if predict_test == 0:
				lbl21.config(text="LR predict ads is unsuitable! (-1)", fg = "black")
			elif predict_test == 1:
				lbl21.config(text="LR predict ads is suitable! (1)", fg = "black")
		elif model_value is 2:
			f = open(root.active_folder_link + '/naive_bayes_classifier.pickle', 'rb')
			NB = pickle.load(f)
			predict_test = NB.predict(new_features) # label is 0 or 1
			f.close()
			if predict_test == 0:
				lbl21.config(text="NB predict ads is unsuitable! (-1)", fg = "black")
			elif predict_test == 1:
				lbl21.config(text="NB predict ads is suitable! (1)", fg = "black")
		elif model_value is 3:
			f = open(root.active_folder_link + '/decision_tree_classifier.pickle', 'rb')
			DT = pickle.load(f)
			predict_test = DT.predict(new_features) # label is 0 or 1
			f.close()
			if predict_test == 0:
				lbl21.config(text="DT predict ads is unsuitable! (-1)", fg = "black")
			elif predict_test == 1:
				lbl21.config(text="DT predict ads is suitable! (1)", fg = "black")
		elif model_value is 4:
			f = open(root.active_folder_link + '/KNN_classifier.pickle', 'rb')
			KNN = pickle.load(f)
			predict_test = KNN.predict(new_features) # label is 0 or 1
			f.close()
			if predict_test == 0:
				lbl21.config(text="KNN predict ads is unsuitable! (-1)", fg = "black")
			elif predict_test == 1:
				lbl21.config(text="KNN predict ads is suitable! (1)", fg = "black")


	lbl = tk.Label(root, text = "Farm-Ads Dataset Classification", bg = "white", fg = "black", font = ('Arial', 18))
	lbl.pack(side='top')

	btn = tk.Button(root, text = "Close", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = close_window)
	btn.place(x=620, y=475, height=22, width=80)


	lbl2 = tk.Label(root, text = "Dataset Link:", bg = "white", fg = "black", font = ('Arial', 11))
	lbl2.place(x=20, y=50)

	btn2 = tk.Button(root, text = "Choose Dataset", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = load_dataset_link )
	btn2.place(x=200, y=50, height=22, width=150)

	lbl3 = tk.Label(root, text = "_dataset link should be here", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl3.place(x=360, y=50)


	lbl4 = tk.Label(root, text = "Train/Test Set Ratio:", bg = "white", fg = "black", font = ('Arial', 11))
	lbl4.place(x=20, y=80)

	ratio_str = tk.StringVar() #string that contain ratio of train/test with format x:y
	entry_train_test_ratio = tk.Entry(root, textvariable = ratio_str)
	entry_train_test_ratio.place(x=200, y=80, height=22, width=150)

	lbl5 = tk.Label(root, text = "_ratio format: x/y", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl5.place(x=360, y=80)


	lbl6 = tk.Label(root, text = "Pre-process Dataset:", bg = "white", fg = "black", font = ('Arial', 11))
	lbl6.place(x=20, y=110)

	btn3 = tk.Button(root, text = "Begin", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = preprocess_data )
	btn3.place(x=200, y=110, height=22, width=150)

	lbl7 = tk.Label(root, text = "_preprocessing status", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl7.place(x=360, y=110)

	lbl8 = tk.Label(root, text = "---------------------------------------------------------------------------------------------------------------------------------------------------------------------", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl8.place(x=20, y=140)

	lbl9 = tk.Label(root, text = "Select Training Method", bg = "white", fg = "black", font = ('Arial', 11))
	lbl9.place(x=20, y=160)

	btn4 = tk.Button(root, text = "Logistic\nRegression", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = Logistic_Regression_button)
	btn4.place(x=50, y=190, height=55, width=90)

	btn5 = tk.Button(root, text = "Naive Bayes", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = Naive_Bayes_button )
	btn5.place(x=50, y=260, height=55, width=90)

	btn6 = tk.Button(root, text = "Decision Tree", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = Decision_Tree_button )
	btn6.place(x=50, y=330, height=55, width=90)

	btn7 = tk.Button(root, text = "K-Nearest\nNeighbors", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = KNN_button )
	btn7.place(x=50, y=400, height=55, width=90)

	n_neighbors = tk.StringVar() #number of neighbors as a param for KNN
	entry_n_neighbors = tk.Entry(root, textvariable = n_neighbors)
	entry_n_neighbors.place(x=150, y=400, height=22, width=22)

	lbl10 = tk.Label(root, text = "Evaluation", bg = "white", fg = "black", font = ('Arial', 11))
	lbl10.place(x=260, y=160)

	lbl11 = tk.Label(root, text = "_", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl11.place(x=200, y=190)

	lbl12 = tk.Label(root, text = "_", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl12.place(x=200, y=260)

	lbl13 = tk.Label(root, text = "_", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl13.place(x=200, y=330)

	lbl14 = tk.Label(root, text = "_", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl14.place(x=200, y=400)


	lbl15 = tk.Label(root, text = "Testing", bg = "white", fg = "black", font = ('Arial', 11))
	lbl15.place(x=525, y=160)

	lbl16 = tk.Label(root, text = "Select Model:", bg = "white", fg = "black", font = ('Arial', 11))
	lbl16.place(x=420, y=180)


	value_checklist = IntVar() #value of checklist input for choosing model for testing data
	checklist1 = tk.Radiobutton(root, text = 'Logistic Regression', variable = value_checklist, value = 1, bg = "white", fg = "black")
	checklist1.place(x=440, y=205)

	checklist2 = tk.Radiobutton(root, text = 'Naive Bayes', variable = value_checklist, value = 2, bg = "white", fg = "black")
	checklist2.place(x=590, y=205)

	checklist3 = tk.Radiobutton(root, text = 'Decision Tree', variable = value_checklist, value = 3, bg = "white", fg = "black")
	checklist3.place(x=440, y=225)

	checklist4 = tk.Radiobutton(root, text = ' K-NN', variable = value_checklist, value = 4, bg = "white", fg = "black")
	checklist4.place(x=590, y=225)


	lbl17 = tk.Label(root, text = "Input Testing Data:", bg = "white", fg = "black", font = ('Arial', 11))
	lbl17.place(x=420, y=250)

	lbl18 = tk.Label(root, text = "1.", bg = "white", fg = "black", font = ('Arial', 10))
	lbl18.place(x=420, y=280)

	btn8 = tk.Button(root, text = "Choose File", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = load_testfile_link)
	btn8.place(x=440, y=280, height=22, width=120)

	btn9 = tk.Button(root, text = "Predict", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = process_multiple_test_input)
	btn9.place(x=580, y=280, height=22, width=120)


	lbl19 = tk.Label(root, text = "2.", bg = "white", fg = "black", font = ('Arial', 10))
	lbl19.place(x=420, y=320)

	test_input = tk.StringVar() #string that contain input data for testing
	entry_test_input = tk.Entry(root, textvariable = test_input)
	entry_test_input.place(x=440, y=320, height=22, width=260)

	lbl20 = tk.Label(root, text = "_type your input", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl20.place(x=440, y=350)

	btn10 = tk.Button(root, text = "Predict Typed Data", bg = "white", fg = "black", width = 20, height = 1, font = ('Arial', 10), command = process_singular_test_input)
	btn10.place(x=580, y=350, height=22, width=120)

	lbl21 = tk.Label(root, text = "_result output should be here!", bg = "white", fg = "gray", font = ('Arial', 10))
	lbl21.place(x=440, y=385)


	def about_info():
		messagebox.showinfo("About", "University of Infomation Technology\nFaculty of Computer Science\nComputer Science Honor Program 2017\n\nFinal Term Project: Data Mining and Applications\n\nGuiders:\n   Assoc. Prof. Nguyen Hoang Tu Anh\n   M. Nguyen Thi Anh Thu\n\nStudents:\n   17520324 - Nguyen Thanh Danh\n   17520828 - Phan Nguyen\n   17521244 - Ho Sy Tuyen")



	menu = Menu(root)
	root.config(menu=menu) 
	filemenu = Menu(menu) 
	menu.add_cascade(label='File', menu=filemenu) 
	filemenu.add_command(label='Exit', command=root.quit) 
	helpmenu = Menu(menu) 
	menu.add_cascade(label='Help', menu=helpmenu) 
	helpmenu.add_command(label='About', command = about_info) 

	root.mainloop()

if __name__ == '__main__':
	main()