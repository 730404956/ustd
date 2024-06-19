import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class STD:
	class Affinity:
		def get_spectral_cosine_sim(self, qry_feats, doc_feats):
			sim_mat = cosine_similarity(doc_feats, qry_feats)
			return sim_mat
		def distance_mat(s, t, dist=cosine_similarity):
			N = s.shape[0]
			M = t.shape[0]

			dist_mat = np.zeros((N, M))

			for i in range(N):
				for j in range(M):
					dist_mat[i, j] = dist(s[i], t[j])
			return dist_mat
		
		def SCA(self, s1, s2, n=30):
			'''
				return spectral correlation between two spectral features 
			'''	
			n_s1_s2 = n*np.sum(np.dot(s1,s2))
			s1_sum = np.sum(s1)
			s2_sum = np.sum(s2)
			numerator = n_s1_s2 - (s1_sum * s2_sum)
			s1_squared_sum = np.sum(np.square(s1))
			s1_sum_squared = np.square(np.sum(s1))
			s2_squared_sum = np.sum(np.square(s2))
			s2_sum_squared = np.square(np.sum(s2))
			denom_a = n*s1_squared_sum-(s1_sum_squared)
			denom_b = n*s2_squared_sum-(s2_sum_squared)
			denom = np.sqrt(denom_a*denom_b)
			corr = numerator/denom
			return corr
		
		def get_spectral_corr_sim(self, qry_feats, doc_feats): 
			'''
				compute the spectral correlation similarity matrix
			'''
			spectrum_size = qry_feats.shape[1] # size of the feature vector
			sim_mat = np.zeros((doc_feats.shape[0], qry_feats.shape[0]))
			# print(qry_feats.shape)
			# print(doc_feats.shape)
			# print(sim_mat.shape)
			for d in range(sim_mat.shape[0]): 
				for q in range(sim_mat.shape[1]): 
					q_q = qry_feats[q,:]
					t_t = doc_feats[d,:]
					sim_mat[d,q] = self.SCA(t_t,q_q,spectrum_size)
			return sim_mat
		
		def compute_heuristic_affinity(self,qry_feats, doc_feats, threshold=0.8): 
			'''
				generate a heuristic matrix for selecting the region of similarity 
			'''
			tau = 0.8 
			corr_threshold = tau
		
			# step 1. Compute the corr similarity meatrix between qry and doc 
			spec_corr_mat = self.get_spectral_corr_sim(qry_feats, doc_feats)
			
			# step 2. Compute the cosine similarity meatrix between qry and doc 
			spec_cosine_mat = self.get_spectral_cosine_sim(qry_feats, doc_feats)
						
			# step 3. apply the thresholding 
			spec_corr_mat = np.where(spec_corr_mat>corr_threshold,1,spec_corr_mat)
			
			# step 4. Compute the hybrid similarity matrix between corr matrix and cosine matrix 
			hybrid_mat = np.zeros((spec_corr_mat.shape[0], spec_corr_mat.shape[1]))
			
			hybrid_mat = np.multiply(spec_corr_mat, spec_cosine_mat)
			# hybrid_mat = spec_corr_mat
			
			#print(hybrid_mat)
			match_score = 1
			
			#hybrid_thresh = round((corr_threshold+cosine_threshold)/2,2)
			hybrid_thresh = corr_threshold
			
			C = np.zeros((spec_corr_mat.shape[0],  spec_corr_mat.shape[1]), int)
			for i in range(0, hybrid_mat.shape[0]): 
				for j in range(0, hybrid_mat.shape[1]): 
					if(hybrid_mat[i,j]>=hybrid_thresh):
						if(i>0 and j>0):
							match =  match_score+C[i - 1, j - 1]
							C[i, j] = match
						elif(i==0 or j==0):
							C[i,j] = match_score
			
			# step 5. trace the diagonal changes through gradient of order 2 
			qry_size = qry_feats.shape[0] # length of query feats 

			max_offset = hybrid_mat.shape[0]  # maximum number of diagonal scans 
			
			h_cost = np.zeros((hybrid_mat.shape[0]))
			
			query_h_cost = qry_size*(qry_size+1)/2 # arithmatic progression of n * n+1 /2
			for i in range(0,max_offset-1): 
				diags = C.diagonal(-i)
				grad = np.array(diags)
				seq_count = np.sum(grad)
				h_cost[i] = round(seq_count/query_h_cost,3)
			
			
			max_index = np.where(h_cost==np.amax(h_cost))[0][0]
			max_indexes = np.where(h_cost==np.amax(h_cost))[0]
			sim_st_index = max_index
			sim_end_index = max_index+qry_size
			return hybrid_mat, h_cost, sim_st_index, sim_end_index
