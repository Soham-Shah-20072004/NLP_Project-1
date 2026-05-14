from .util import *
import math

class Evaluation():
	def _get_true_doc_IDs(self, query_id, qrels):
		ids=set()
		for thing in qrels:
			if int(thing["query_num"])==int(query_id):
				ids.add(int(thing["id"]))
		return ids
	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		if k<=0: return 0.0
		topk=query_doc_IDs_ordered[:k]
		actual=set(int(d) for d in true_doc_IDs)
		got=sum(1 for x in topk if int(x) in actual)
		return got/k
	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		vals=[]
		for i,qid in enumerate(query_ids):
			rel=self._get_true_doc_IDs(qid, qrels)
			if len(rel)==0: continue
			vals.append(self.queryPrecision(doc_IDs_ordered[i], qid, rel, k))
		return sum(vals)/len(vals) if vals else 0.0
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		actual=set(int(d) for d in true_doc_IDs)
		if len(actual)==0: return 0.0
		topk=query_doc_IDs_ordered[:k]
		got=sum(1 for x in topk if int(x) in actual)
		return got/len(actual)
	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		vals=[]
		for i,qid in enumerate(query_ids):
			rel=self._get_true_doc_IDs(qid, qrels)
			if len(rel)==0: continue
			vals.append(self.queryRecall(doc_IDs_ordered[i], qid, rel, k))
		return sum(vals)/len(vals) if vals else 0.0
	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		b=0.5
		p=self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		r=self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		bottom=(b**2)*p+r
		return 0.0 if bottom==0 else (1+b**2)*p*r/bottom
	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		vals=[]
		for i,qid in enumerate(query_ids):
			rel=self._get_true_doc_IDs(qid, qrels)
			if len(rel)==0: continue
			vals.append(self.queryFscore(doc_IDs_ordered[i], qid, rel, k))
		return sum(vals)/len(vals) if vals else 0.0
	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		actual=set(int(d) for d in true_doc_IDs)
		if len(actual)==0: return 0.0
		dcg=sum((1 if int(doc) in actual else 0)/math.log2(i+2) for i,doc in enumerate(query_doc_IDs_ordered[:k]))
		idcg=sum(r/math.log2(i+2) for i,r in enumerate([1]*min(len(actual),k)+[0]*max(0,k-len(actual))))
		return 0.0 if idcg==0 else dcg/idcg
	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		vals=[]
		for i,qid in enumerate(query_ids):
			rel=self._get_true_doc_IDs(qid, qrels)
			if len(rel)==0: continue
			vals.append(self.queryNDCG(doc_IDs_ordered[i], qid, rel, k))
		return sum(vals)/len(vals) if vals else 0.0
	def queryAveragePrecision(self, retrieved_doc_IDs, true_doc_IDs):
		if len(true_doc_IDs)==0: return 0.0
		cnt=0
		total=0.0
		for pos,doc in enumerate(retrieved_doc_IDs, start=1):
			if doc in true_doc_IDs:
				cnt+=1
				total+=cnt/pos
		return total/len(true_doc_IDs)
	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels):
		aps=[]
		for i,qid in enumerate(query_ids):
			rel=self._get_true_doc_IDs(qid, q_rels)
			if len(rel)==0: continue
			aps.append(self.queryAveragePrecision(doc_IDs_ordered[i], rel))
		return 0.0 if len(aps)==0 else sum(aps)/len(aps)
	def queryReciprocalRank(self, query_doc_IDs_ordered, true_doc_IDs):
		actual=set(int(d) for d in true_doc_IDs)
		for pos,doc in enumerate(query_doc_IDs_ordered, start=1):
			if int(doc) in actual: return 1.0/pos
		return 0.0
	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels):
		rrs=[]
		for i,qid in enumerate(query_ids):
			rel=self._get_true_doc_IDs(qid, qrels)
			if len(rel)==0: continue
			rrs.append(self.queryReciprocalRank(doc_IDs_ordered[i], rel))
		return 0.0 if len(rrs)==0 else sum(rrs)/len(rrs)
