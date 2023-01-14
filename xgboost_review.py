import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import re
from bs4 import BeautifulSoup

def review_to_wordlist(review):
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    #用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 小写化所有的词，并转成词list
    #return review_text
    words = review_text.lower().split()
    # 返回words
    return words


stops = set(stopwords.words("english"))      
meaningful_words = [w for w in words if not w in stops] 

from sklearn.feature_extraction.text import CountVectorizer
# bag of words tool
vectorizer = CountVectorizer(analyzer = "word",
                          tokenizer = None,
                          preprocessor = None,
                          stop_words = 'english',
                          max_features = 5000)
train_x = vectorizer.fit_transform(train_data)
train_x = train_x.toarray()
print 'bag of words处理结束！'


from xgboost import XGBClassifier
#from xgboost import XGBRegressor
xgbc = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=500,
                     subsample=0.8,
                     colsample_btree=0.8,
                     objective='binary:logitraw')

### 交叉验证
#from sklearn.cross_validation import cross_val_score
#scores = cross_val_score(xgbc, train_x, label, cv=3, scoring='accuracy')
#print scores
#print scores.mean()

xgbc = XGBClassifier(max_depth=3 learning_rate=0.1, n_estimators=300,
                     subsample=0.8,
                     colsample_btree=0.8,
                     objective='binary:logitraw')


fmap = dict(map(lambda x: ('f%d' % (x[1]), x[0]), tfidf.vocabulary_.items()))
import xgboost as xgb
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.025,
    'alpha': 1,
    'seed': 0,
    #'nthread': 8,
    'silent': 1
}

# 留出法
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, label, test_size=0.2, random_state=31)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain,'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist)
ypred = bst.predict(dtest)
y_pred = (ypred >= 0.5) * 1
from sklearn import metrics
print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)
print 'ACC: %.4f' % metrics.accuracy_score(y_test, y_pred)
print 'Recall: %.4f' % metrics.recall_score(y_test, y_pred)
print 'F1-score: %.4f' %metrics.f1_score(y_test, y_pred)
print 'Precesion: %.4f' %metrics.precision_score(y_test, y_pred)
for key, value in sorted(bst.get_fscore().items(), key=lambda x: -x[1])[:20]:
    print fmap.get(key), value

dtest = xgb.DMatrix(test_x)
ypred = bst.predict(dtest)
test_predicted = (ypred >= 0.5) * 1

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
tfidf = TFIDF(min_df=2, # 最小支持度为2
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词
# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)
tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print 'TF-IDF处理结束.'
# # 去掉单个字母的元素
#     words = filter(lambda x: len(x) >= 2, words)