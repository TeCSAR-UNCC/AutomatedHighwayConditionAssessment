#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


wl = pd.read_csv('classifier_results.csv', header=None)
wl.columns=['true','wrong','conf']


# In[3]:


wl


# In[4]:


wl=wl.drop('conf', 1)
wl['true'] = wl['true'].str.replace('Original label:','')
wl['wrong'] = wl['wrong'].str.replace('Prediction :','')


# In[5]:


wl


# In[6]:


wl.true.unique()


# In[7]:


wl.wrong.unique()


# In[8]:


wl.true.count()


# In[9]:


wl.wrong.count()


# In[10]:


ctg=wl.groupby(['true', 'wrong']).size()
ctg.to_csv('countgroup.csv')
ctgh = pd.read_csv('countgroup.csv', header=None)


# In[11]:


ctgh


# In[12]:


#Starting to generate Confusion table
wl['true'] = wl[['true', 'wrong']].apply(lambda x: ''.join(x), axis=1)
wl=wl.drop('wrong', 1)
wl.columns=['truewrong']
ct = wl.groupby('truewrong').size()
ct.to_csv('count.csv')


# In[13]:


pivoted = ctgh.pivot(0,1)
pivoted=pivoted.fillna(0)
pivoted


# In[14]:


#DR50N=99;DR55N=142;DR56N=77;DR57N=58;DR58N=74;RO60N=47;RO61N=121;RO65N=141;RS40N=84;RS41N=51;RS42N=111;TR11N=59;TR13N=57;TR24N=148
DR50N=180;DR55N=666;DR57N=182;DR58N=99;RO60N=340;RO65N=300;RS40N=339;RS41N=99;RS42N=403;TR11N=120;TR13N=611;TR24N=658


# In[15]:


pivoted[0:1]=pivoted[0:1]/DR50N
pivoted[1:2]=pivoted[1:2]/DR55N
#pivoted[2:3]=pivoted[2:3]/DR56N
pivoted[2:3]=pivoted[2:3]/DR57N
pivoted[3:4]=pivoted[3:4]/DR58N
pivoted[4:5]=pivoted[4:5]/RO60N
#pivoted[6:7]=pivoted[6:7]/RO61N
pivoted[5:6]=pivoted[5:6]/RO65N
pivoted[6:7]=pivoted[6:7]/RS40N
pivoted[7:8]=pivoted[7:8]/RS41N
pivoted[8:9]=pivoted[8:9]/RS42N
pivoted[9:10]=pivoted[9:10]/TR11N
pivoted[10:11]=pivoted[10:11]/TR13N
pivoted[11:12]=pivoted[11:12]/TR24N


# In[16]:


pivoted.to_csv('pivoted.csv')
pivoted


# In[17]:


#Normalized final confusion table
piv=pd.read_csv('pivoted.csv', header=None)
piv=piv.drop(0, 0)
piv=piv.drop(2,0)
piv=piv.drop(1,0)
piv=piv.drop(0,1)
piv = piv.apply(pd.to_numeric)


# In[18]:


piv


# Generating the heatmap confusion matrix

# In[19]:


piv=piv[[1,2,3,5,6,7,9,11,12,4,8,10]]
piv=piv.reindex([3,4,5,7,8,9,11,13,14,6,10,12])
labels=['Small pipes & Box culverts','Ditch','Under-Edge drains','Brush & Tree','Slope','Flexible pavements','Paved shoulders',
        'Pavement markers','Object markers & Delineators', 'Storm drains & drop inlets','Rigid pavement','Signs']
pivmat=piv.as_matrix(columns=None)
plt.figure(figsize=(12, 9))
fig=sns.heatmap(pivmat, linecolor='blue', cmap='viridis', annot=True)
a=fig.xaxis.set_ticklabels(labels,rotation=85)
b=fig.yaxis.set_ticklabels(labels,rotation=0)
plt.savefig('conf_matrix.png', bbox_inches='tight')


# In[20]:


#Pivoted sum
pivsum=pivoted
pivsum.sum(axis=1)


# In[21]:


#Per class accuracy
DR50N=180;DR55N=666;DR57N=182;DR58N=99;RO60N=340;RO65N=300;RS40N=339;RS41N=99;RS42N=403;TR11N=120;TR13N=611;TR24N=658
DR50acc=(DR50N-111)/DR50N
DR55acc=(DR55N-106)/DR55N
DR57acc=(DR57N-54)/DR57N

RO60acc=(RO60N-59)/RO60N
RO65acc=(RO65N-113)/RO65N
RS40acc=(RS40N-78)/RS40N

RS42acc=(RS42N-91)/RS42N

TR13acc=(TR13N-98)/TR13N
TR24acc=(TR24N-110)/TR24N

DR58acc=(DR58N-72)/DR58N
RS41acc=(RS41N-53)/RS41N
TR11acc=(TR11N-52)/TR11N


# # Plotting per class accuracy for 12 classes

# In[22]:


#objects = ('DR50','DR55','DR57','RO60','RO65','RS40','RS42','TR13','TR24','DR58','RS41','TR11')
objects = ('Small pipes & Box culverts','Ditch','Under-Edge drains','Brush & Tree','Slope','Flexible pavements','Paved shoulders',
        'Pavement markers','Object markers & Delineators', 'Storm drains & drop inlets','Rigid pavement','Signs')
y_pos = np.arange(len(objects))
performance = [DR50acc, DR55acc, DR57acc, RO60acc, RO65acc, RS40acc, RS42acc, TR13acc, TR24acc, DR58acc, RS41acc, TR11acc]
width=0.5
plt.figure(figsize=(10,5))
plt.bar(y_pos, performance, width, align='center', alpha=0.95)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Accuracy')
plt.title('Accuracy per class')
plt.show()


# # Plotting misclassification bar charts for classes with less than 60% accuracy

# In[23]:


objects = ('Ditch','Under-Edge Drains','Paved Shoulders','Object markers & Delineators')
y_pos = np.arange(len(objects))
performance = [36/99,6/99,8/99,10/99]
width=0.3
plt.figure(figsize=(8,6))
plt.bar(y_pos, performance, width, align='center', alpha=0.95)
plt.xticks(y_pos, objects, rotation=80, fontsize=15)
plt.ylabel('Error')
plt.title('Storm Drains & Drop Inlets-Wrong prediction per class')
plt.show()


# In[24]:


objects = ('Flexible Pavement','Paved Shoulders','Pavement Markers','Ditch', 'Object markers & Delineators')
y_pos = np.arange(len(objects))
performance = [23/99,13/99,7/99,4/99,4/99]
width=0.3
plt.figure(figsize=(8,6))
plt.bar(y_pos, performance, width, align='center', alpha=0.95)
plt.xticks(y_pos, objects, rotation=80, fontsize=15)
plt.ylabel('Error')
plt.title('Rigid Pavements-Wrong prediction per class')
plt.show()


# In[25]:


objects = ('Ditch', 'Brush & Tree', 'Slope','Object markers & Delineators')
y_pos = np.arange(len(objects))
performance = [13/120,4/120,5/120,21/120]
width=0.3
plt.figure(figsize=(8,6))
plt.bar(y_pos, performance, width, align='center', alpha=0.95)
plt.xticks(y_pos, objects, rotation=80, fontsize=15)
plt.ylabel('Error')
plt.title('Signs-Wrong prediction per class')
plt.show()


# In[26]:


objects = ('Ditch', 'Under-Edge Drains','Slope')
y_pos = np.arange(len(objects))
performance = [62/180,39/180,5/180]
width=0.3
plt.figure(figsize=(8,6))
plt.bar(y_pos, performance, width, align='center', alpha=0.95)
plt.xticks(y_pos, objects, rotation=80, fontsize=15)
plt.ylabel('Error')
plt.title('Small Pipes & Box Culverts-Wrong prediction per class')
plt.show()



