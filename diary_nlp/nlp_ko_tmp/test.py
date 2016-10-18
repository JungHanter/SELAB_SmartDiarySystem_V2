import xlrd
from pandas import DataFrame
import pickle
import numpy as np
import csv
from pandas import DataFrame

# matrix = []
#
# f = open('./v.2.csv','r')
# csvReader = csv.reader(f)
#
# for row in csvReader:
#     matrix.append(row[0].split())
#
# f.close()
# word = []
# list = []
# tmp = []
#
# for row in matrix:
#     word.append(row[0])
#     for idx, c in enumerate(row):
#         if idx == 0:
#             if len(row) == 1:
#                 tmp.append(None)
#             continue
#         tmp.append(c)
#     list.append(tmp)
#     tmp = []
#
# # for w, l in zip(word, list):
# #     print("w : %s %s"%(w, l))
#
# dt = DataFrame({'word': word, 'collections': list}, columns=['word', 'collections'])
# print(dt)
# with open('collection', 'wb') as f:   # train set 저장 하는 코드
#     pickle.dump(dt, f)
with open('collection', 'rb') as f:  #저장한 pickle 불러오기
    collection = pickle.load(f)

print(collection)

# table_1 = xlrd.open_workbook('table_1.xlsx')
# table_2 = xlrd.open_workbook('table_2.xlsx')
#
# idxs_1 = []
# idxs_2 = []
# words = []
# std_idx = []
# word_idx = []
# for sheet in table_1.sheets():
#     for i in range(1, sheet.nrows):
#         row = sheet.row_values(i)
#         idxs_1.append(int(row[0]))
#         words.append(row[1])
# #
# for sheet in table_2.sheets():
#     for i in range(1, sheet.nrows):
#         row = sheet.row_values(i)
#         idxs_2.append(int(row[0]))
#         word_idx.append(int(row[1]))
#         std_idx.append(int(row[2]))
#
# #
# dt_word = DataFrame({'idx': idxs_1, 'words': words})
# dt_corr = DataFrame({'idx': idxs_2, 'word_idx': word_idx, 'std_idx': std_idx})
#
# with open('tb2', 'wb') as f:   # train set 저장 하는 코드
#     pickle.dump(dt_corr, f)
# #
# with open('tb1', 'rb') as f1, open('tb2', 'rb') as f2:  #저장한 pickle 불러오기
#     dt_word = pickle.load(f1)
#     dt_corr = pickle.load(f2)
#
# list = np.array(dt_word)
# print("list : ", list)
# print("dt_word : ", dt_word)
# print("dt_corr : ", dt_corr)