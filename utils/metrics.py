import numpy as np
import math


class SaliencyMetrics():
    '''Function to calculate metrincs for saliency map evaluation
    AUC_borji, AUC_shuffled, nss, cc
    '''

    @staticmethod
    def normalize_map(s_map):
        norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
        return norm_s_map

    def auc_borji(self,s_map,gt,splits=100,stepsize=0.1,gt_threshold=255/2):
        gt = np.where(gt < gt_threshold, 0, 1)
        num_fixations = int(np.sum(gt))

        num_pixels = s_map.shape[0]*s_map.shape[1]
        random_numbers = np.random.randint(0,num_pixels,(splits,num_fixations))

        aucs = []
        # for each split, calculate auc
        for i in random_numbers:
            r_sal_map = []
            for k in i:
                r_sal_map.append(s_map[k%s_map.shape[0]-1, k//s_map.shape[0]])
            # in these values, we need to find thresholds and calculate auc
            thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

            r_sal_map = np.array(r_sal_map)

            thresholds = sorted(set(thresholds))
            area = []
            area.append((0.0,0.0))
            for thresh in thresholds:
                # in the salience map, keep only those pixels with values above threshold
                temp = np.zeros(s_map.shape)
                temp[s_map>=thresh] = 1.0
                num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
                tp = num_overlap/(num_fixations*1.0)
                
                fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

                area.append((round(tp,4),round(fp,4)))

            area.append((1.0,1.0))
            area.sort(key = lambda x:x[0])
            tp_list =  [x[0] for x in area]
            fp_list =  [x[1] for x in area]

            aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))

            return np.mean(aucs)

    def auc_shuff(self,s_map,gt,other_map,splits=100,stepsize=0.1,gt_threshold=255/2):
        gt = np.where(gt < gt_threshold, 0, 1)
        other_map = np.where(other_map < gt_threshold, 0, 1)

        num_fixations = np.sum(gt)

        x,y = np.where(other_map==1)
        other_map_fixs = []
        for j in zip(x,y):
            other_map_fixs.append(j[0]*other_map.shape[0] + j[1])
        ind = len(other_map_fixs)
        assert ind==np.sum(other_map), 'something is wrong in auc shuffle'


        num_fixations_other = min(ind,num_fixations)

        num_pixels = s_map.shape[0]*s_map.shape[1]
        random_numbers = []
        for i in range(0,splits):
            temp_list = []
            t1 = np.random.permutation(ind)
            for k in t1:
                temp_list.append(other_map_fixs[k])
            random_numbers.append(temp_list)	

        aucs = []
        # for each split, calculate auc
        for i in random_numbers:
            r_sal_map = []
            for k in i:
                r_sal_map.append(s_map[k%s_map.shape[0]-1, k//s_map.shape[0]])
            # in these values, we need to find thresholds and calculate auc
            thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

            r_sal_map = np.array(r_sal_map)

            thresholds = sorted(set(thresholds))
            area = []
            area.append((0.0,0.0))
            for thresh in thresholds:
                # in the salience map, keep only those pixels with values above threshold
                temp = np.zeros(s_map.shape)
                temp[s_map>=thresh] = 1.0
                num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
                tp = num_overlap/(num_fixations*1.0)
                
                fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

                area.append((round(tp,4),round(fp,4)))
            
            area.append((1.0,1.0))
            area.sort(key = lambda x:x[0])
            tp_list =  [x[0] for x in area]
            fp_list =  [x[1] for x in area]

            aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))

        return np.mean(aucs)

    @staticmethod
    def nss(s_map,gt,gt_threshold=255/2):
        gt = np.where(gt < gt_threshold, 0, 1)
        s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)

        x,y = np.where(gt==1)
        temp = []
        for i in zip(x,y):
            temp.append(s_map_norm[i[0],i[1]])
        return np.mean(temp)

    @staticmethod
    def cc(s_map,gt,gt_threshold=255/2):
        gt = np.where(gt < gt_threshold, 0, 1)
        s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
        gt_norm = (gt - np.mean(gt))/np.std(gt)
        a = s_map_norm
        b= gt_norm
        r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
        return r


    def calculate_metrics(self,s_map_list,gt_list):
        auc_b = []
        auc_s = []
        nss_list = []
        cc_list = []

        for idx in range(len(gt_list)):
            auc_b.append(self.auc_borji(self.normalize_map(s_map_list[idx]),gt_list[idx]))
            auc_s.append(self.auc_shuff(self.normalize_map(s_map_list[idx]),gt_list[idx],gt_list[idx]))
            nss_list.append(self.nss(s_map_list[idx],gt_list[idx]))
            cc_list.append(self.cc(s_map_list[idx],gt_list[idx]))

        return np.mean(auc_b), np.mean(auc_s), np.mean(nss_list), np.mean(cc_list)