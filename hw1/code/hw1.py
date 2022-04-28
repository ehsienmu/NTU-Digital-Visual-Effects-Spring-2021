import os
import glob
import cv2
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import math
import random
import sys
import numpy as np
import argparse
from tqdm import tqdm

def load(path_test):
    filenames = []
    ln_exposure_times = []
    f = open(os.path.join(path_test, 'image_list.txt'),'r')
    for line in f.readlines():
        #print("line",line)
        if (line[0] == '#'):
            continue
        
        (filename, exposure) = line.split()
        filenames += [os.path.join(path_test,filename)]
        ln_exp = np.log(float(exposure))
        ln_exposure_times += [ln_exp]

    return filenames, ln_exposure_times

def read(path_list):
    #print("path_list[0]",path_list[0])

    shape = cv2.imread(path_list[0]).shape  #3456*5184*3
    compress = 1
    if shape[0] > 2000:
        compress = 4
    #print("Image shape",shape)
    img_list_4d = np.zeros((len(path_list), shape[0]//compress, shape[1]//compress, 3), dtype=float)
    for i in path_list:
        img = cv2.imread(i)
        resImg = cv2.resize(img,(img.shape[1]//compress,img.shape[0]//compress),interpolation = cv2.INTER_CUBIC)
        img_list_4d[path_list.index(i), :, :, :] = resImg
    return img_list_4d

def weight(Z):
    #print("Z----",Z)
    Z_min, Z_max = 0., 256.
    if Z <= (Z_min + Z_max) /2:
        w = Z - Z_min
    else:
        w = Z_max - Z
    return w

def sample(img_list_4d): #random sample from 
    Z_min, Z_max = 0, 256

    num_images = len(img_list_4d)
    num_sample = Z_max - Z_min
    sample = np.zeros((num_sample, num_images), dtype=float)
   
    sample_img = img_list_4d[2]
    idx_list = []
    for i in range(Z_min, Z_max):
        rows, cols  = np.where(sample_img == i)
        if len(rows) == 0:
            continue
        idx = random.randrange(len(rows))
        for j in range(num_images):
            sample[i, j] = img_list_4d[j][rows[idx], cols[idx]]
    return sample

def exp(my_list):
    return [ math.exp(x) for x in my_list]

def log(my_list):
    return [ math.log(x) for x in my_list]

#equation 3
def Debevec_response_curve(Z, B, l): #Z exp 100
    Z_min, Z_max = 0., 255.

    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0] + 1), dtype=float)
    b = np.zeros((A.shape[0], 1), dtype=float)
    #print("Z.shape",Z.shape)
    #print("B.shape",len(B))

    #Include the data fitting equation
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            w_ij = weight(int(Z[i,j]+1))
            #print("w_ij",w_ij)
            A[k, int(Z[i,j]+1)] = w_ij
            A[k, n + i] = -w_ij
            b[k, 0] = w_ij * B[j]
            k += 1
    #print("k",k)
    #Fix the curve by setting its middle value to 0
    A[k, int((Z_max - Z_min) // 2)] = 1
    k = k+1
    #print("k",k)
    #Include the smoothness equations
    for q in range(n-1):
        w_q = weight(q+1)
        A[k, q-1] = l * w_q
        A[k, q] = -2 * l * w_q
        A[k, q+1] = l * w_q
        k = k+1
    #print("k",k)
    #solve the system using SVD
    x = np.dot(np.linalg.pinv(A), b)
    g = x[0 : n]
    lE = x[n : x.shape[0]]
    #print("g--------",g.shape)
    #print("lE",lE.shape)
    return g, lE

def Robertson_response_curve(img_list_4d, B ,routine = 4):
    delta_t = exp(B)
    #print("delta_t",len(delta_t),delta_t)
    Z = np.array([img.flatten().tolist() for img in img_list_4d])
    
    g = (np.arange(0, 1, 1/256))
    #print("Z",Z)
    num_images = len(img_list_4d)
    #print("num_images",num_images)
    for j in range(num_images):
        Z[j] = [int(i) for i in Z[j]]
    pixels = len(Z[0])
    #print("pixels",pixels)
    k = 0
    #print("g",g)
    E_i = np.zeros(len(Z[0]), dtype=float)
    while(k <= routine):
        #print("----------------",k)
        #calculate E_i
        k +=1
        #print("g",g)
        for i in tqdm(range(pixels),desc='Robertson',ascii=True):
            #if(i % 10000 == 0):
            #    print("iiiiiiii",i)
            temp_a = 0
            temp_b = 0
            for j in range(num_images):
                Z_ij = Z[j][i]
                temp_a += weight(Z_ij)*g[int(Z_ij)]*delta_t[j]
                temp_b += weight(Z_ij)*delta_t[j]*delta_t[j]
            E_i[i] = temp_a/temp_b
        #calculate g
        for m in range(257):
            #print("m",m)
            temp_c = 0
            sumNum = 0
            
            num,pos = np.where(Z == m)
            if(len(pos) == 0):
                continue
            for q in range(len(pos)):
                #print("pos[q]",pos[q])
                temp_c += E_i[pos[q]] * delta_t[num[q]]
            #print("temp_c",temp_c)
            #print("E_i[pos[q]]",E_i[pos[q]])
            #sumNum += len(pos)
            #print("temp_c/sumNum",temp_c//sumNum)
            #print("g[m]",g[m])
            g[m] = temp_c/len(pos)
         
    #print("Z",len(Z))
    #print("g",len(g),g)
    return log(g), E_i

#equation 6
def compute_Radiance(img_list_4d, g, ln_exps):

    img_size = img_list_4d[0].shape
    # print("img_size",img_size)
    Z = [img.flatten().tolist() for img in img_list_4d]
    # print("img_list_4d_shape",len(Z[0])," ",len(Z))

    ln_E = np.zeros(len(Z[0]), dtype=float)
    #temp = np.zeros(len(Z[0]), dtype=np.float64)
    pixels = len(Z[0])
    num_images = len(img_list_4d)
    
    # print("num_images",num_images)
    for i in tqdm(range(pixels),desc='computing radiance', ascii=True):
        # if(i % 100000 == 0):
        #     print("iiiiiiii",i)
       
        sumW = 0
        temp = 0
        for j in range(num_images):
            Z_ij = Z[j][i]
            temp += weight(Z_ij) * (g[int(Z_ij)] - ln_exps[j])
            sumW += weight(Z_ij)
            
        
        ln_E[i] = temp / sumW
    # print("ln_E",ln_E)    
    E = exp(ln_E)  
    #print("ln_E",E)
    ln_hdr = np.reshape(ln_E, img_size)
    hdr = np.reshape(E, img_size)
    return ln_hdr,hdr

def Drago_tonemapping(hdr,mul=1.2):
    tonemapDrago = cv2.createTonemapDrago(1.3, 1.0)
    ldrDrago = tonemapDrago.process(hdr.astype('float32'))
    ldrDrago = (ldrDrago * mul)
    return ldrDrago * 255

# def Drago_tonemapping(hdr,mul):
#     tonemapDrago = cv2.createTonemapDrago(1.3, 1.0)
#     ldrDrago = tonemapDrago.process(hdr.astype('float32'))
#     ldrDrago = ldrDrago * mul
#     cv2.imwrite("tonemapping.jpg", ldrDrago * 255)

def get_lm(lw,a, sigma):
    lw_bar = np.exp(np.average(np.log(sigma+lw)))
    lm = a*(lw)/lw_bar
    return lm

def get_ld_global(lm,l_white):
    return lm * (1 + ((lm) / (l_white) / (l_white))) / (1 + lm)

def get_ldr_global(hdr_image, a=1, sigma=0.0000001, l_white=0.5):
    lw = hdr_image
    lm = get_lm(lw, a, sigma)
    ld = get_ld_global(lm, l_white)
    ld = lm / (1 + lm)
    ldr = (ld * 255).astype(int)
    return ldr

def get_l_blur(lm, s):
    return cv2.GaussianBlur(lm, (2*s + 1, 2*s + 1), 0)

def get_ldr_local(hdr_image,a=1,sigma=0.0000001,eps=0.5,phi=2):
    lw = hdr_image
    lm = get_lm(lw,a,sigma)
    blur_all = np.zeros((9,)+hdr_image.shape)
    for i in range(0,9):
        blur_all[i] = get_l_blur(lm,i)
    vs = np.zeros((7,)+(hdr_image.shape))
    for i in range(0,7):
        vs[i] = np.abs((blur_all[i+1]  - blur_all[i+2])/((2**phi)*a/((i+1)**2) + blur_all[i+1]))
    smax_arr = np.zeros([hdr_image.shape[0],hdr_image.shape[1]],int)
    ld = np.zeros(hdr_image.shape)
    for x in range(0,hdr_image.shape[0]):
        for y in range(0,hdr_image.shape[1]):
            smax= max(0,np.argmin(np.average(vs[:,x,y,:],axis=1) >= eps)-1)
            smax_arr[x,y]=smax+1
            ld[x,y]=lm[x,y]/(1+blur_all[smax+1,x,y,:])
    ldr = ((ld*255).astype(int))
    return ldr

def gray_image(image):
    gray_img = np.zeros([image.shape[0], image.shape[1]],'int')
    for r in range(0,image.shape[0]):
        for c in range(0,image.shape[1]):
            #print(c.shape)
            R, G, B = int(image[r][c][2]), int(image[r][c][1]), int(image[r][c][0])
            y = round((54 * R + 183 * G + 19 * B)/255)
            
            gray_img[r][c] = y
    return gray_img

def shrink_img(image, ratio):
    while ratio!=1:
        image = np.delete(image, range(1, image.shape[0], 2), axis=0)
        image = np.delete(image, range(1, image.shape[1], 2), axis=1)
        ratio = ratio // 2
    return image

def median_array(arr,ignore=0):
    return np.full(arr.shape,np.median(arr)-ignore)
    
def filter_median(arr,ignore = 0):
    return ((arr < (median_array(arr,ignore))).astype(int)[:] * 255)

def compare_img(imga,imgb):
    return (np.count_nonzero(imga != imgb))

def shift_img(img, x_shift, y_shift): #想成A不動動B為基準
    #print('img.shape,x_shift, y_shift = ',img.shape, x_shift, y_shift)
    #xshift
    if x_shift!=0:
        img = (img[:, 0:-(x_shift)] if x_shift > 0 else img[:, -(x_shift):])
    #yshift
    if y_shift!=0:
        img = (img[y_shift:, :] if y_shift > 0 else img[0:(y_shift),: ])

    #print('img.shape becomes ',img.shape)
    return img

def shift_compare(imga, imgb, bx_shift, by_shift): #想成A不動動B為基準
    #print("imga in shift compare = ", imga.shape)
    imga = shift_img(imga, -bx_shift, -by_shift)
    imgb = shift_img(imgb, bx_shift, by_shift)
    return compare_img(filter_median(imga), filter_median(imgb))

def get_bestshift(imga, imgb):
    best_x, best_y = 0, 0
    min_diff_cnt = 1000000
    for x in range(-1,2):
        for y in range(-1,2):
            diff_cnt = shift_compare(imga, imgb, x, y)
            if diff_cnt < min_diff_cnt:
                min_diff_cnt = diff_cnt
                best_x, best_y = x, y
    #print('best = ', best_x, best_y)
    return best_x, best_y

def alignment_twoimg(imga, imgb):
    # gray1 = cv2.cvtColor(imga.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(imgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    gray1 = cv2.cvtColor(imga.astype(np.float32), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(imgb.astype(np.float32), cv2.COLOR_BGR2GRAY)
    imga_32 = shrink_img(gray1, 32)
    imgb_32 = shrink_img(gray2, 32) #邊長變成32分之1
    xs,ys=get_bestshift(imga_32,imgb_32)
    imga_16 = shift_img(shrink_img(gray1, 16),-2*xs,-2*ys)
    imgb_16 = shift_img(shrink_img(gray2, 16),2*xs,2*ys)
    xx,yy=get_bestshift(imga_16,imgb_16)
    xs=2*xs+xx
    ys=2*ys+yy
    
    imga_8 = shift_img(shrink_img(gray1, 8),-2*xs,-2*ys)
    imgb_8 = shift_img(shrink_img(gray2, 8),2*xs,2*ys)
    xx,yy=get_bestshift(imga_8,imgb_8)
    xs=2*xs+xx
    ys=2*ys+yy
    
    imga_4 = shift_img(shrink_img(gray1, 4),-2*xs,-2*ys)
    imgb_4 = shift_img(shrink_img(gray2, 4),2*xs,2*ys)
    xx,yy=get_bestshift(imga_4,imgb_4)
    xs=2*xs+xx
    ys=2*ys+yy
    
    imga_2 = shift_img(shrink_img(gray1, 2),-2*xs,-2*ys)
    imgb_2 = shift_img(shrink_img(gray2, 2),2*xs,2*ys)
    xx,yy=get_bestshift(imga_2,imgb_2)
    xs=2*xs+xx
    ys=2*ys+yy
    
    imga_0 = shift_img(gray1,-2*xs,-2*ys)
    imgb_0 = shift_img(gray2,2*xs,2*ys)
    xx,yy=get_bestshift(imga_0,imgb_0)
    xs=2*xs+xx
    ys=2*ys+yy
    
    imga_0 = shift_img(imga_0,-xs,-ys)
    imgb_0 = shift_img(imga_0,xs,ys)
    
    imga_00 = shift_img(gray1,-xs,-ys)
    imgb_00 = shift_img(gray2,xs,ys)
    x_shift=xs
    y_shift=ys
    #print('final shift = ', x_shift, y_shift)
    return x_shift, y_shift

def get_move_list(img_list):
    xs = 0
    ys = 0
    move = []
    move_r = []
    for i in range(len(img_list) - 1): #遍歷資料夾
        image_1 = img_list[i]
        image_2 = img_list[i + 1]
        #if(fi!=0):
        #    b_xrs, b_yrs = alignment_twoimg(image_1[2000:2450,2385:2757], image_2[2000:2450,2385:2757])
        #else:
        b_xrs, b_yrs = alignment_twoimg(image_1, image_2)
        #break
        xs += b_xrs
        ys += b_yrs
        move_r.append(([i + 1], (b_xrs, b_yrs)))
        move.append((xs,ys))
        # img = np.roll(image_2,xs,axis=1)
        # img = np.roll(img,-ys,axis=0)
        # cv2.imwrite(path+'align_'+files[fi+1],img[9:-9,12:-12])   
    return move

def MTB_alignment(img_list):
    move_list = get_move_list(img_list)
    # print('move_list', move_list)
    abs_move = [(abs(number[0]),abs(number[1])) for number in move_list]
    # print('abs_move', abs_move)
    max_x_mv =  max(abs_move,key=lambda item:item[0])[0]
    max_y_mv =  max(abs_move,key=lambda item:item[1])[1]
    # print('max x mv', max_x_mv)
    # print('max y mv', max_y_mv)
    xmv = 0
    ymv = 0
    yxratio= img_list[0].shape[0]/img_list[0].shape[1]
    if max_y_mv >= max_x_mv:
        ymv = max_y_mv
        xmv = int(ymv / yxratio)
    else:
        xmv = max_x_mv
        ymv = int(xmv*yxratio)
    # print(xmv,ymv)
    # print(img_list[0].shape)
    if(xmv!=0 and ymv!=0):
        new_img_list=np.zeros([img_list.shape[0],img_list[0][ymv:-ymv,xmv:-xmv].shape[0],img_list[0][ymv:-ymv,xmv:-xmv].shape[1],3])
        new_img_list[0] = img_list[0][ymv:-ymv,xmv:-xmv]
        cv2.imwrite('alignment_0.jpg',img_list[0][ymv:-ymv,xmv:-xmv])
    else:
        new_img_list = np.zeros(img_list.shape)
        new_img_list[0] = img_list[0]
        cv2.imwrite('alignment_0.jpg',img_list[0])
    for i in range(1,img_list.shape[0]):
        xs,ys = move_list[i-1]
        img = np.roll(img_list[i],xs,axis=1)
        img = np.roll(img,-ys,axis=0)
        if(xmv!=0 and ymv!=0):
            new_img_list[i] = img[ymv:-ymv,xmv:-xmv]
            cv2.imwrite('alignment_'+str(i)+'.jpg',img[ymv:-ymv,xmv:-xmv])   
        else:
            new_img_list[i] = img
            cv2.imwrite('alignment_'+str(i)+'.jpg',img)
    return new_img_list

#main function
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Run some functions')

    parser.add_argument('-r', '--radiance_method', help='Radiance method', dest='radiance_method', default='Debevec')
    parser.add_argument('-t', '--tone_mapping_method', help='Tone mapping method method', dest='tone_mapping_method', default='Global')
    parser.add_argument('input_image_file', help='Images directory path')
    #parser.add_argument('-o', '--output', help='Output filename', dest='output_file',)
    parser.add_argument('-mul', help='mul parameter in Drago', dest='mul', default='1.1')
    parser.add_argument('-a', help='a parameter in Global/Local', dest='a', default='1.0')
    parser.add_argument('-sigma', help='sigma parameter in Local', dest='sigma', default='0.000001')
    parser.add_argument('-e', '-epsilon', help='epsilon parameter in Local', dest='eps', default='0.004')
    parser.add_argument('-phi', help='phi parameter in Local', dest='phi', default='8')
    parser.add_argument('-w', help='l_white in Global', dest='l_white', default='1.5')
    args = parser.parse_args()
    print('input argument = ', args)
    if(args.input_image_file):
        img_file = args.input_image_file
    
    img_file, output_file,  = sys.argv[1],'main_hallway.hdr'
    print("Load pictures")

    if(img_file == 'cwd'):
        #print('os.getcwd():',os.getcwd())
        img_file = os.getcwd()
    img_list, ln_exps = load(img_file)
    img_list_4d = read(img_list) 

    print("Align pictures")
    #print("img_list_4d[0].shape",img_list_4d[0].shape)
    img_list_4d = MTB_alignment(img_list_4d)
    #print("MTB_img_list_4d[0].shape",img_list_4d[0].shape)
    
    img_count = 0

    layer_img_list_4d_r = []
    layer_img_list_4d_g = []
    layer_img_list_4d_b = []
   
    for img in img_list_4d:
        layer_img_list_4d_r.append(img[:,:,0]) 
        layer_img_list_4d_g.append(img[:,:,1])
        layer_img_list_4d_b.append(img[:,:,2])
        
    Z_r = sample(layer_img_list_4d_r)
    Z_g = sample(layer_img_list_4d_g)
    Z_b = sample(layer_img_list_4d_b)

    hdr = np.zeros((img_list_4d.shape[1], img_list_4d.shape[2], 3), dtype=float)
    ln_hdr = np.zeros((img_list_4d.shape[1], img_list_4d.shape[2], 3), dtype=float)

    if(args.radiance_method == 'Debevec'):
        print("Debevec")
        g_r, lE_r = Debevec_response_curve(Z_r, ln_exps, 100.)
        #print("lE_r",lE_r)
        #print("g_r",g_r)
        g_g, lE_g = Debevec_response_curve(Z_g, ln_exps, 100.)
        #print("lE_g",lE_g)
        g_b, lE_b = Debevec_response_curve(Z_b, ln_exps, 100.)
        #print("lE_b",lE_b)

    elif(args.radiance_method == 'Robertson'):
        print("Robertson")
        g_r, E_r = Robertson_response_curve(layer_img_list_4d_r, ln_exps)
        g_g, E_g = Robertson_response_curve(layer_img_list_4d_g, ln_exps)
        g_b, E_b = Robertson_response_curve(layer_img_list_4d_b, ln_exps)

        
        
    
    print('Compute Radiance: Red')
    ln_hdr_r ,hdr_r = compute_Radiance(np.array(layer_img_list_4d_r), g_r, ln_exps)
    print('Compute Radiance: Green')
    ln_hdr_g,hdr_g = compute_Radiance(np.array(layer_img_list_4d_g), g_g, ln_exps)
    print('Compute Radiance: Blue')
    ln_hdr_b,hdr_b = compute_Radiance(np.array(layer_img_list_4d_b), g_b, ln_exps)

    #hdr_img[:,:, c] = cv2.normalize(img_rad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #hdr = np.zeros((mg_list_4d.shape[1], img_list_4d.shape[2], 3), 'float32')
    
    
    plt.figure(figsize=(10, 10))
    plt.plot(g_r, range(256), 'rx')
    plt.plot(g_g, range(256), 'gx')
    plt.plot(g_b, range(256), 'bx')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig('response-curve.png')
    print("response-curve")
    hdr[:,:,0] = hdr_r
    hdr[:,:,1] = hdr_g
    hdr[:,:,2] = hdr_b
    #print(hdr)

    # with open('hdr.npy','wb')as f:
    #     np.save(f,hdr)
    # hdr = np.load('hdr.npy')
    
    cv2.imwrite('hdr_img.hdr', hdr)
    # k = cv2.imread('hdr_img.hdr')

    # print(type(k))

    plt.figure(figsize=(10,8))
    plt.imshow(np.log(cv2.cvtColor(hdr.astype('float32'), cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig('radiance-map-library.png')
    print("radiance-map")

    if (args.tone_mapping_method == 'Drago'):
        ldr = Drago_tonemapping(hdr,float(args.mul))
        cv2.imwrite(str(args.input_image_file)+"\\tonemapping_drago.jpg", ldr)
        print("tonemapping by Drago")

    elif (args.tone_mapping_method == 'Global'):
        ldr = get_ldr_global(hdr,a=float(args.a),sigma=float(args.sigma),l_white=float(args.l_white))
        cv2.imwrite(str(args.input_image_file)+"\\tonemapping_global.jpg", ldr)
        print("tonemapping by Global Operator")

    elif (args.tone_mapping_method == 'Local'):
        ldr_image_local = get_ldr_local(hdr,a=float(args.a),sigma=float(args.sigma),eps=float(args.eps),phi=float(args.phi))
        cv2.imwrite(str(args.input_image_file)+"\\tonemapping_local.jpg", ldr)
        print("tonemapping by Local Operator")


    # print("Tonemapping")
    # print("Drago")
    # cv2.imwrite(str(args.input_image_file)+"\\tonemapping_drago.jpg",  Drago_tonemapping(hdr,float(args.mul)))
    # print("Global")
    # cv2.imwrite(str(args.input_image_file)+"\\tonemapping_global.jpg", get_ldr_global(hdr,a=float(args.a),sigma=float(args.sigma),l_white=float(args.l_white)))
    # print("Local")
    # cv2.imwrite(str(args.input_image_file)+"\\tonemapping_local.jpg", get_ldr_local(hdr,a=float(args.a),sigma=float(args.sigma),eps=float(args.eps),phi=float(args.phi)))
