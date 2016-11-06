
# coding: utf-8

# In[1]:

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')


# In[2]:

np.set_printoptions(precision=4)


# In[3]:

# g === area (big image)
# f === fragment
g = Image.open('img/20160825-143207-156-1.jpg').convert('L')

wg = g.width // 4
hg = g.height // 4
xg = g.width // 2 - wg // 2
yg = g.width // 2 - hg // 2

box = (xg, yg, xg + wg, yg + hg)
g = g.crop(box)
g


# In[4]:

print g.size
g_mat = np.asarray(g.getdata(), dtype=np.int8).reshape(g.size[1], g.size[0])


# In[5]:

# Crop fragment from area image
xf = 300
yf = 50
wf = 200
hf = 200

crop_box = (xf, yf, xf + wf, yf + hf)
f = g.crop(crop_box)
print f.size
f


# In[6]:

# Create matrix of fragment
f_mat = np.asarray(f.getdata(), dtype=np.int8).reshape(f.size[1], f.size[0])
# Flip matrix
f_mat = np.fliplr(f_mat)
f_mat = np.flipud(f_mat)

# Image.fromarray(f_mat, 'L')


# In[7]:

num_shades = 256
# Create indicators of f
# of size == g.size
chi = np.zeros((num_shades, g.size[1], g.size[0]), dtype=bool)


# In[8]:

# fill the indicators
for h in xrange(f.size[1]):
    for w in xrange(f.size[0]):
        color = f_mat[h, w]
        chi[color, h, w] = True


# In[9]:

# Image.fromarray(chi.sum(axis=0).astype('uint8')*(num_shades - 1), 'L')


# In[10]:

# chi_elems[i] === number of pixels that have color "i"
chi_elems = np.array( f.histogram() )
print chi_elems


# In[11]:

fft_chi = np.fft.fft2(chi)


# In[ ]:

fft_g = np.fft.fft2(g_mat)


# In[ ]:

# Scalar product (g_frag, chi[i])
sp_g_frag_chi = np.zeros((num_shades, g.size[1] - hf, g.size[0] - wf))

for i in xrange(num_shades):
    if chi_elems[i] > 0:
        sp_g_frag_chi[i] = np.fft.ifft2(fft_g * fft_chi[i])[hf:, wf:]


# In[ ]:

# || Projection of g_frag on f ||^2
norm_pr_gfrag_sqr = np.zeros((g.size[1] - hf, g.size[0] - wf))
for i in xrange(num_shades):
    if chi_elems[i] > 0:
        norm_pr_gfrag_sqr += sp_g_frag_chi[i] ** 2 / float(chi_elems[i])
        

norm_pr_gfrag_sqr = abs(norm_pr_gfrag_sqr)


# In[ ]:

# plt.plot(norm_pr_gfrag_sqr.ravel())


# In[ ]:

w_norm = norm_pr_gfrag_sqr.shape[1]
idx_pr = norm_pr_gfrag_sqr.argmax()
true_idx = xf + yf * w_norm
print 'idx Should be: ', true_idx
print idx_pr
print idx_pr // w_norm + 1, idx_pr % w_norm + 1


# In[ ]:

# plt.plot(np.arange(true_idx - 1000, true_idx + 1000), norm_pr_gfrag_sqr.ravel()[true_idx - 1000 : true_idx + 1000])


# In[ ]:

# chi_X -- const field of vision
# 1 1 1 0 0 ... 0
# 1 1 1 0 0 ... 0
# 1 1 1 0 0 ... 0
# 0 0 0 0 0 ... 0
# . . .
# 0 0 0 0 0 ... 0
chi_X = np.zeros((g.size[1], g.size[0]), dtype=bool)
chi_X[:hf, :wf] = np.ones((hf, wf))

# || g ||^2
fft_gsqr = np.fft.fft2(g_mat ** 2)
fft_chi_X = np.fft.fft2(chi_X)
norm_gfrag_sqr = np.fft.ifft2(fft_gsqr * fft_chi_X)[hf:, wf:].astype('float')

norm_gfrag_sqr = abs(norm_gfrag_sqr)


# In[ ]:

# plt.plot(norm_gfrag_sqr.ravel())


# In[ ]:

# E_gfrag = np.fft.ifft2(fft_g * fft_chi_X)[hf:, wf:].astype('float')
norm_E_gfrag_sqr = np.fft.ifft2(fft_g * fft_chi_X)[hf:, wf:].astype('float')                         ** 2 / (hf * wf)

norm_E_gfrag_sqr = abs(norm_E_gfrag_sqr)


# In[ ]:

# plt.plot(norm_E_gfrag_sqr.ravel())


# In[ ]:

numerator = norm_gfrag_sqr - norm_pr_gfrag_sqr
plt.plot(numerator.ravel()).savefig("numer.png")


# In[ ]:

denominator = norm_pr_gfrag_sqr - norm_E_gfrag_sqr
plt.plot(denominator.ravel()).savefig("denom.png")


# In[ ]:

w_norm = norm_pr_gfrag_sqr.shape[1]
idx_denom = denominator.ravel().argmax()
true_idx = xf - 1 + (yf - 1) * w_norm
print 'idx Should be: ', true_idx
print idx_denom
y_denom, x_denom = idx_denom // w_norm + 1, idx_denom % w_norm + 1

print y_denom, x_denom
print yf, xf

# Image.fromarray(g_mat[y_denom : y_denom + hf, x_denom : x_denom + wf], 'L')


# In[ ]:

f


# In[ ]:

tau = abs(numerator) / abs(denominator)
# plt.plot(tau.ravel())


# In[ ]:

index = tau.argmin() 
x_min = index % tau.shape[1] + 1
y_min = index // tau.shape[1] + 1
print index
print 'Should be:', true_idx
print "x_min, y_min: %d %d" % (y_min, x_min)
print 'Should be:', (yf, xf)


# In[ ]:

# Image.fromarray(g_mat[y_min : y_min + hf , x_min : x_min + wf], 'L')


# In[ ]:



