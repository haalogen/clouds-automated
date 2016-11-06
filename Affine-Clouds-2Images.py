
# coding: utf-8

# In[133]:

#!/usr/bin/env python

# import sys
# import os
import numpy as np
from PIL import Image

np.set_printoptions(suppress=True)


# In[134]:

# A) =Вход=
# 1) 2 файла изображений стереопары (в некой папке img/):
# 20160824-174253-406-1.jpg
# 20160824-174253-406-2.jpg

date = "20160909-141139-078"

# for ipynb
fname_left = 'img/' + date + '-1.jpg'
fname_right = 'img/' + date + '-2.jpg'

# for cmd line run
# fname_left = os.path.abspath(sys.argv[0])
# fname_right = os.path.abspath(sys.argv[1])


# In[135]:

img_left = Image.open(fname_left).convert(mode='L')
img_right = Image.open(fname_right).convert(mode='L')
print """Images loaded as grayscale:
%s
%s""" % (fname_left, fname_right)


# In[136]:

# 2) Конфигурация эксперимента
# Txt-файлы (в папке config) 
# * Аффинные+дисторсные коэффициенты для цифровой юстировки стереопары:
# файл aff_dist.txt: a, b, c, d, e, f, eps1, eps2 -- 8 коэффициентов

# rX = a*lX + b*lY + e - eps1*z_x(lX, lY) + eps2*z_x(rX, rY)
# rY = c*lX + d*lY + f - eps1*z_y(lX, lY) + eps2*z_y(rX, rY),
# where approximately(!):
# z_x = (x-x0)*[ (x-x0)^2 +(y-y0)^2 ] = z_x(rX, rY) = z_x(lX, lY)
# z_y = (y-y0)*[ (x-x0)^2 +(y-y0)^2 ] = z_y(rX, rY) = z_y(lY, lY)

align_coeffs = np.loadtxt('config/aff_dist.txt')

print 'Align coeeficients:\n', align_coeffs


# In[137]:

# B) Алгоритм автоматизированного анализа стереопары

# a) Подготовка к анализу:
# -- Юстировка("Нормализация") изображений для возможности анализа.
a = align_coeffs[0];
b = align_coeffs[1];
c = align_coeffs[2];
d = align_coeffs[3];
e = align_coeffs[4];
f = align_coeffs[5];
eps1 = align_coeffs[6];
eps2 = align_coeffs[7];


# In[138]:

det = a * d - b * c;
inv_a = d / det;
inv_b = -b / det;
inv_c = -c / det;
inv_d = a / det;


# In[139]:

def affine_transform_point(x, y):
    return [b * y + x * a + e , d * y + x * c + f]


# In[140]:

def apply_affine(img_left, img_right):
    width = img_left.width
    height = img_left.height
    
    aff_coord = np.zeros((4, 2))
#     affine transformation of the corner points
    aff_coord[0] = affine_transform_point(0, 0)
    aff_coord[1] = affine_transform_point(width, 0)
    aff_coord[2] = affine_transform_point(0, height)
    aff_coord[3] = affine_transform_point(width, height)
    
#     the rightmost (biggest by value) x-coordinate of the transformed
#     left-top and left-bottom x-coordinates
    x0 = int( max(aff_coord[0, 0], aff_coord[2, 0]) )
#     the lowermost (biggest by value) y-coordinate of the transformed
#     left-top and right-top y-coordinates
    y0 = int( max(aff_coord[0, 1], aff_coord[1, 1]) )
#     the leftmost (smallest by value) x-coordinate of the transformed
#     right-top and right-bottom x-coordinates
    x1 = int( min(aff_coord[1, 0], aff_coord[3, 0]) )
#     the uppermost (smallest by value) y-coordinate of the transformed
#     left-bottom and right-bottom y-coordinates
    y1 = int( min(aff_coord[2, 1], aff_coord[3, 1]) )
    
#     n_x0 -- x-coordinate of the new left-bot point
    n_x0 = int( max(0, x0) )
#     n_y0 -- y-coordinate of the new left-bot point
    n_y0 = int( max(0, y0) )
#     n_x1 -- x-coordinate of the new right-top point
    n_x1 = int( min(width, x1) )
#     n_y1 -- y-coordinate of the new right-top point
    n_y1 = int( min(height, y1) )
    
    nw = n_x1 - n_x0 # new width
    nh = n_y1 - n_y0 # new height
    
    new_left_img = Image.new(mode='L', size=(nw, nh))
    new_right_img = Image.new(mode='L', size=(nw, nh))
    
    # Load pixmaps
    l_pix = img_left.load()
    r_pix = img_right.load()
    nl_pix = new_left_img.load()
    nr_pix = new_right_img.load()
    
    
    
    for  y in xrange(n_y0, n_y1):
        for x in xrange(n_x0, n_x1):
# Let's calculate backwards our original coordinates of the left image
            orig_x = int( (x - e) * inv_a + (y - f) * inv_b )
            orig_y = int( (x - e) * inv_c + (y - f) * inv_d )
            
#             assert(0 <= orig_x <= width)
#             assert(0 <= orig_y <= height)
            
# paint new images with coordinates from (0,0) to (nw - 1, nh - 1)
            nl_pix[x - n_x0, y - n_y0] = l_pix[orig_x, orig_y]
            nr_pix[x - n_x0, y - n_y0] = r_pix[x, y]
    
    return (new_left_img, new_right_img)


# In[141]:

img_left_n, img_right_n = apply_affine(img_left, img_right)

img_left = img_left_n
img_right = img_right_n


# In[142]:

fname_left[:-4]+"_aff_applied.png"


# In[143]:

img_left.save(fname_left[: -4] + "_aff_applied.png")
img_right.save(fname_right[: -4] + "_aff_applied.png")

