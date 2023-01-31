#Name - Sayem Lincoln
#PID - A54207835
#CSE 404 Homework 10


# coding: utf-8

# In[21]:


import scipy.io
mat = scipy.io.loadmat('USPS.mat.mat')



data,label=mat['A'],mat['L']


# In[26]:


# data.shape


# In[27]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(data)


# In[28]:


# X_std.shape


# In[29]:


import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#print('Covariance matrix \n%s' %cov_mat)


# In[30]:


# cov_mat.shape


# In[31]:


#print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# In[32]:


cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)


# In[33]:


# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)


# In[48]:


cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[35]:


u,s,v = np.linalg.svd(X_std.T)
# u


# In[36]:


for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))


# In[37]:


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[46]:


import chart_studio.plotly as py
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,30)],
    y=var_exp,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,30)], 
    y=cum_var_exp,
    name='Cumulative'
)

data = [trace1, trace2]

layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)

fig = dict(data=data, layout=layout)
# from  plotly.offline import plot
import chart_studio.plotly as py
import plotly.graph_objs as go
# these two lines allow your code to show up in a notebook
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
iplot(fig, filename='selecting-principal-components')


# In[47]:


# matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
#                       eig_pairs[1][1].reshape(4,1)))

# print('Matrix W:\n', matrix_w)

