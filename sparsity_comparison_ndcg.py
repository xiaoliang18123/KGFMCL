import matplotlib.pyplot as plt

# 构造数据
x = [1, 2, 3, 4]
last_fm = ['<62', '<135', '<290', '<2881']
yelp2018 = ['<16', '<32', '<78', '<2057']
alibaba_fashion = ['<12', '<17', '<32', '<393']
# mf
# last_fm
last_fm_mf_recall = [0.09599, 0.03338, 0.02045, 0.01133]
last_fm_mf_ndcg = [0.07131, 0.04123, 0.04990, 0.06619]
# yelp2018
yelp2018_mf_recall = [0.07233, 0.06572, 0.05696, 0.04045]
yelp2018_mf_ndcg = [0.03938, 0.04310, 0.04840, 0.06050]
# alibaba-fashion
alibaba_fashion_mf_recall = [0.11597, 0.10807, 0.0997, 0.08293]
alibaba_fashion_mf_ndcg = [0.06600, 0.06648, 0.06965, 0.07402]
# kgat
# last_fm
last_fm_kgat_recall = [0.11667, 0.03577, 0.02043, 0.01103]
last_fm_kgat_ndcg = [0.08789, 0.04388, 0.05028, 0.06340]
# yelp2018
yelp2018_kgat_recall = [0.07552, 0.06800, 0.06073, 0.04264]
yelp2018_kgat_ndcg = [0.04153, 0.04551, 0.05100, 0.06387]
# alibaba-fashion
alibaba_fashion_kgat_recall = [0.11343, 0.10643, 0.09800, 0.08099]
alibaba_fashion_kgat_ndcg = [0.06491, 0.06553, 0.06854, 0.07318]
# kgin
# last_fm
last_fm_kgin_recall = [0.12961, 0.03881, 0.02291, 0.01347]
last_fm_kgin_ndcg = [0.09978, 0.04892, 0.05514, 0.07855]
# yelp2018
yelp2018_kgin_recall = [0.07748, 0.07432, 0.06455, 0.04930]
yelp2018_kgin_ndcg = [0.04265, 0.05012, 0.05548, 0.07472]
# alibaba-fashion
alibaba_fashion_kgin_recall = [0.12092, 0.11413, 0.10544, 0.08933]
alibaba_fashion_kgin_ndcg = [0.07010, 0.07103, 0.07526, 0.08068]
# kgcl
# last_fm
# last_fm_kgcl_recall = [0.07732, 0.02111, 0.01166, 0.00592]
# last_fm_kgcl_ndcg = [0.05771, 0.04251, 0.05341, 0.06875]
last_fm_kgcl_recall = [0.12167, 0.03677, 0.02103, 0.01203]
last_fm_kgcl_ndcg = [0.09389, 0.04588, 0.05278, 0.07140]
# yelp2018
yelp2018_kgcl_recall = [0.08038, 0.07650, 0.06650, 0.04739]
yelp2018_kgcl_ndcg = [0.04512, 0.05100, 0.05637, 0.07210]
# alibaba-fashion
alibaba_fashion_kgcl_recall = [0.11840, 0.11306, 0.10480, 0.08806]
alibaba_fashion_kgcl_ndcg = [0.06876, 0.07118, 0.07547, 0.07985]
# KGRec
# last_fm
last_fm_kgrec_recall = [0.12847, 0.03755, 0.02173, 0.01302]
last_fm_kgrec_ndcg = [0.09778, 0.04695, 0.05314, 0.07645]
# yelp2018
yelp2018_kgrec_recall = [0.07861, 0.07452, 0.06465, 0.04911]
yelp2018_kgrec_ndcg = [0.04325, 0.04912, 0.05586, 0.07132]
# alibaba-fashion
alibaba_fashion_kgrec_recall = [0.12492, 0.11806, 0.10953, 0.09081]
alibaba_fashion_kgrec_ndcg = [0.07198, 0.07376, 0.07763, 0.08379]
# kgcna
# last_fm
last_fm_kgcna_recall = [0.13297, 0.03855, 0.02193, 0.01362]
last_fm_kgcna_ndcg = [0.10092, 0.04745, 0.05414, 0.08063]
# yelp2018
yelp2018_kgcna_recall = [0.07944, 0.07364, 0.06491, 0.04566]
yelp2018_kgcna_ndcg = [0.04415, 0.04882, 0.05497, 0.06924]
# alibaba-fashion
alibaba_fashion_kgcna_recall = [0.12772, 0.1207, 0.11159, 0.09490]
alibaba_fashion_kgcna_ndcg = [0.07378, 0.07538, 0.07913, 0.08552]
# kgccl
# last_fm
last_fm_kgcncl_recall = [0.13028, 0.03998, 0.02426, 0.01384]
last_fm_kgcncl_ndcg = [0.10564, 0.05026, 0.05861, 0.08044]
# yelp2018
yelp2018_kgcncl_recall = [0.08828, 0.08332, 0.07100, 0.04960]
yelp2018_kgcncl_ndcg = [0.04975, 0.05610, 0.06023, 0.07739]
# alibaba-fashion
alibaba_fashion_kgcncl_recall = [0.13863, 0.13261, 0.12146, 0.10287]
alibaba_fashion_kgcncl_ndcg = [0.08210, 0.08478, 0.08889, 0.09589]
# 创建画布和子图
fig1, axs1 = plt.subplots(1, 3, figsize=(12, 4))

# 画第一个子图
# 设置每个子图的横坐标标签
axs1[0].set_xticks(x)
axs1[0].set_xticklabels(last_fm)
axs1[0].plot(x, last_fm_mf_ndcg, label='MF')
axs1[0].plot(x, last_fm_kgat_ndcg, label='KGAT')
axs1[0].plot(x, last_fm_kgin_ndcg, label='KGIN')
axs1[0].plot(x, last_fm_kgcl_ndcg, label='KGCL', color='SaddleBrown')
axs1[0].plot(x, last_fm_kgrec_ndcg, label='KGRec', color='Black')
axs1[0].plot(x, last_fm_kgcna_ndcg, label='KGCNA', color='purple')
axs1[0].plot(x, last_fm_kgcncl_ndcg, label='KGFMCL', color='red')
axs1[0].set_title('Last-FM')
axs1[0].set_xlabel('User Group')
axs1[0].set_ylabel('ndcg@20')
axs1[0].legend()

# 画第二个子图
# 设置每个子图的横坐标标签
axs1[1].set_xticks(x)
axs1[1].set_xticklabels(yelp2018)
axs1[1].plot(x, yelp2018_mf_ndcg, label='MF')
axs1[1].plot(x, yelp2018_kgat_ndcg, label='KGAT')
axs1[1].plot(x, yelp2018_kgin_ndcg, label='KGIN')
axs1[1].plot(x, yelp2018_kgcl_ndcg, label='KGCL', color='SaddleBrown')
axs1[1].plot(x, yelp2018_kgrec_ndcg, label='KGRec', color='Black')
axs1[1].plot(x, yelp2018_kgcna_ndcg, label='KGCNA', color='purple')
axs1[1].plot(x, yelp2018_kgcncl_ndcg, label='KGFMCL', color='red')
axs1[1].set_title('Yelp2018')
axs1[1].set_xlabel('User Group')
axs1[1].set_ylabel('ndcg@20')
axs1[1].legend()

# 画第三个子图
# 设置每个子图的横坐标标签
axs1[2].set_xticks(x)
axs1[2].set_xticklabels(alibaba_fashion)
axs1[2].plot(x, alibaba_fashion_mf_ndcg, label='MF')
axs1[2].plot(x, alibaba_fashion_kgat_ndcg, label='KGAT')
axs1[2].plot(x, alibaba_fashion_kgin_ndcg, label='KGIN')
axs1[2].plot(x, alibaba_fashion_kgcl_ndcg, label='KGCL', color='SaddleBrown')
axs1[2].plot(x, alibaba_fashion_kgrec_ndcg, label='KGRec', color='Black')
axs1[2].plot(x, alibaba_fashion_kgcna_ndcg, label='KGCNA', color='purple')
axs1[2].plot(x, alibaba_fashion_kgcncl_ndcg, label='KGFMCL', color='red')
axs1[2].set_title('Alibaba-iFashion')
axs1[2].set_xlabel('User Group')
axs1[2].set_ylabel('ndcg@20')
axs1[2].legend()
# 设置整体标题和布局
# fig.suptitle('Multiple Line Charts with Multiple Lines in Each Subplot')
plt.tight_layout()
plt.savefig('sparsity_comparison_ndcg.eps', format='eps')
plt.savefig('sparsity_comparison_ndcg.pdf', format='pdf')
# 显示图表
plt.show()
