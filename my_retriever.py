from collections import Counter
import math
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index  
        self.term_weighting = term_weighting 
        self.num_docs = self.compute_number_of_documents()

        self.df = self.get_df()
        self.idf = self.get_idf()
        self.doc_vec_length = self.get_doc_vec_length()
        
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    
    #Gets document frequnecy for each term
    def get_df(self):
        df = {}
        for term in self.index:
            #Number of documents containing the term
            df[term] = len(self.index[term])
        return df
    
    #Gets inverse document frequency for each term
    def get_idf(self):
        idf = {}
        for term, df in self.df.items():
            #Log of the total number of documents/document frequency 
            idf[term] = math.log(self.num_docs/df)
        return idf

    #Gets the magnitude of the document vector based on the term weighting scheme 
    def get_doc_vec_length(self):
        doc_vec_length= {}
        for term in self.index:
            for docid, tf in self.index[term].items():
                #Initialise document vector of a new document
                if docid not in doc_vec_length:
                    doc_vec_length[docid] = 0
                #Compute weight of the term depending on weighting scheme 
                if self.term_weighting == "binary":
                    #If binary weighting, weight is 1
                    weight = 1
                elif self.term_weighting == "tf":
                    #If tf weighting, weight is raw term frequency
                    weight = tf
                elif self.term_weighting == "tfidf":
                    #If tfidf weighting, weight is tf*idf
                    weight = tf*self.idf[term]
                #Accumulate the sum of squared term weights for each document vector 
                doc_vec_length[docid] += weight**2
                    
        for docid in doc_vec_length:
            #Take the sqaure root of the sum of squared term weights for each document vector 
            doc_vec_length[docid] = math.sqrt(doc_vec_length[docid])
        #Return magnitudes of document vectors 
        return doc_vec_length
    
    #Gets the query vector based on the term weighting scheme 
    def get_query_vector(self, q_tf):
        query_vector = {}
        for term, tf in q_tf.items():
            #Compute query term weight depending on weighting scheme
            if self.term_weighting == "binary":
                q_weight = 1
            elif self.term_weighting == "tf":
                q_weight = tf
            elif self.term_weighting == "tfidf":
                idf_value = self.idf.get(term)
                #Check if query term exists in the index (idf)
                if idf_value is not None:
                    q_weight = tf*idf_value
                else: 
                    #If the query term doesn't exist in the index (idf), contributes nothing
                    q_weight = 0
            #Stores the weight in the query vector 
            query_vector[term] = q_weight

        return query_vector


    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        #Stores term frequencies of the query terms in a query
        q_tf = Counter(query)
        #Collect documents that contain least one term in the query 
        candidate_docs = set()
        for term in q_tf:
            #If a query term exists in the index, add document ids containing the term
            if term in self.index:
                candidate_docs.update(self.index[term].keys())
        #If the query term doesn't exist in any document, returm an empty list
        if not candidate_docs:
            return []
        #Gets the query vector 
        query_vector = self.get_query_vector(q_tf)
        scores = {}

        #Compute cosine similarity score for each candidate document 
        for docid in candidate_docs:
            doc_vec_length = self.doc_vec_length.get(docid, 0)
            #Skip documents with zero document vector magnitude 
            if doc_vec_length == 0:
                continue 
            dot = 0
            #Compute dot product between query vector and document vector 
            for term, q_weight in query_vector.items():
                #Ignores 0 weights 
                if q_weight == 0:
                    continue
                #Skip if term doesn't exist in the index
                if term not in self.index:
                    continue
                #Skip if document doesn't contain the term
                if docid not in self.index[term]:
                    continue 
                #Get the term frequency of the term in the current document
                doc_tf = self.index[term][docid]

                #Compute document term weight based on the weighting scheme 
                if self.term_weighting == "binary":
                    d_weight = 1
                elif self.term_weighting == "tf":
                    d_weight = doc_tf
                elif self.term_weighting == "tfidf":
                    d_weight = doc_tf*self.idf[term]
                #Accumulate dot product between query and document vector 
                dot += q_weight*d_weight
            
            #Only store positive dot product score to prevent none relevant documents being returned
            if dot > 0:
                """
                Calculate cosine similarity score by normalising with document vector magnitude. 
                The query vector magnitude is omitted because it is constant for a given query.
                """
                scores[docid] = dot/doc_vec_length

        #Rank the top 10 documents by descending cosine similarity scores
        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:10]
        #Extract document ids from the ranked list
        ranked_docs = [docid for docid, _ in ranked]
        #Return ranked list of document ids in the order of most relevant to least relevant
        return ranked_docs

