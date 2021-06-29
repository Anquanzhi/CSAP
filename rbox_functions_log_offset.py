# -*- coding:UTF-8 -*-
import numpy as np
import ctypes
from ctypes import *
import math

so = ctypes.cdll.LoadLibrary
librbox = so("./librbox.so")
overlap = librbox.Overlap
overlap.argtypes = (POINTER(c_double),POINTER(c_double))
overlap.restype = c_double
DecodeAndNMS = librbox.DecodeAndNMSnp
DecodeAndNMS.argtypes = (POINTER(c_double),POINTER(c_int),POINTER(c_double),POINTER(c_int),c_double)
DecodeAndNMS.restype = None
NMS = librbox.NMS_airplane
NMS.argtypes=(POINTER(c_double),POINTER(c_int),POINTER(c_double),POINTER(c_int),c_double)
NMS.restype=None
stepsize = 4
FEA_WID, FEA_HEI = 75, 75
IM_WID, IM_HEI = 300, 300    
   
def MatchRBox(loc_offset, groundtruth, IM_WIDTH, IM_HEIGHT):
	ind = []
	ind_one_hot = np.zeros((FEA_WID*FEA_HEI))
	for i in range(len(groundtruth)):
		ind_one_hot[int(groundtruth[i][1]*FEA_HEI)*FEA_WID + int(groundtruth[i][0]*FEA_WID)] = 1
		ind.append(int(groundtruth[i][1]*FEA_HEI)*FEA_WID + int(groundtruth[i][0]*FEA_WID))
		if loc_offset:
		    h_offset = (groundtruth[i][1]*FEA_HEI - int(groundtruth[i][1]*FEA_HEI)) / stepsize
		    w_offset = (groundtruth[i][0]*FEA_WID - int(groundtruth[i][0]*FEA_WID)) / stepsize
		    groundtruth[i][:2] = w_offset, h_offset
	if not loc_offset:
	    groundtruth = groundtruth[:, 2:]		    
	indice = np.where(ind_one_hot > 0)[0]
	groundtruth = groundtruth[np.argsort(ind)]
	return ind_one_hot, indice, groundtruth
    

	
def DecodeNMS(loc_preds_j, conf_preds_j, inputloc_j, index, nms_threshold, heightOut, widthOut):
    prior_var = [1.0, 1.0, 0.2, 0.2, 1.0]
    rbox = []
    score = []
    if len(loc_preds_j) > 0:
        loc_c = (c_double * (len(loc_preds_j)/3*5))()
        conf_c = (c_double * len(conf_preds_j))()
        indices_c = (c_int * len(index))()
        loc_preds_j = loc_preds_j.reshape(-1, 5)
        decoded = np.zeros((len(loc_preds_j), 5))
        #decoded[:, 2] = (loc_preds_j[:,0] * prior_var[2]) * widthOut
        decoded[:, 2] = pow(np.exp(1), (loc_preds_j[:,2] * prior_var[2]))/300.0
        decoded[:, 3] = pow(np.exp(1), (loc_preds_j[:,3] * prior_var[3]))/300.0
        decoded[:, 4] = 180.0*(loc_preds_j[:,4] * prior_var[4])
        inds = index.copy()
        decoded[:, 1] = (loc_preds_j[:,1]*stepsize + inds/FEA_WID)*stepsize/300.0
        decoded[:, 0] = (loc_preds_j[:,0]*stepsize + inds%FEA_WID)*stepsize/300.0 #(inds%(widthOut/4))*4/300.0  #inds%(widthOut/4), int(inds/FEA_WID)
        for k in range(len(index)):
            loc_c[5*k+0] = c_double(decoded[k][0])
            loc_c[5*k+1] = c_double(decoded[k][1])
            loc_c[5*k+2] = c_double(decoded[k][2])
            loc_c[5*k+3] = c_double(decoded[k][3])
            loc_c[5*k+4] = c_double(decoded[k][4])
            indices_c[k] = c_int(-1)
            conf_c[k] = c_double(conf_preds_j[k])
        pind = cast(indices_c, POINTER(c_int))
        pconf = cast(conf_c, POINTER(c_double))
        num_preds = c_int(len(index))
        DecodeAndNMS(loc_c, pind, pconf, byref(num_preds), c_double(nms_threshold))		
        for k in range(num_preds.value):
            index_k = indices_c[k]
            area = loc_c[5*index_k + 2] * loc_c[5*index_k + 3] * heightOut * widthOut
            if area < 100 or area > 10000:
                continue
            rbox.append(loc_c[5*index_k] * widthOut * inputloc_j[2] + inputloc_j[0])
            rbox.append(loc_c[5*index_k + 1] * heightOut * inputloc_j[2] + inputloc_j[1])
            rbox.append(loc_c[5*index_k + 2] * widthOut * inputloc_j[2])
            rbox.append(loc_c[5*index_k + 3] * heightOut * inputloc_j[2])
            rbox.append(loc_c[5*index_k + 4])
            score.append(conf_c[index_k])
        return rbox, score    
  

			
    
def NMSOutput(rboxlist, scorelist, nms_threshold, label, test_rbox_output_path):
    loc_c = (c_double * len(rboxlist))()
    score_c = (c_double * len(scorelist))()
    indices_c = (c_int * len(scorelist))()
    for i in range(len(rboxlist)):
        loc_c[i] = c_double(rboxlist[i])
    for i in range(len(scorelist)):
        score_c[i] = c_double(scorelist[i])
        indices_c[i] = c_int(-1)
    num_preds = c_int(len(scorelist))
    NMS(loc_c, indices_c, score_c, byref(num_preds), c_double(nms_threshold))
    with open(test_rbox_output_path, 'w') as fid:
        for i in range(num_preds.value):
            index_i = indices_c[i]
            fid.write('{} {} {} {} {} {} {}\n'.format(loc_c[5*index_i], loc_c[5*index_i+1], loc_c[5*index_i+2], loc_c[5*index_i+3], label,
                       loc_c[5*index_i+4], score_c[index_i]))
        