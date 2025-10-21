"""Different minimiser methods"""

import math
import numpy as np

data = np.array(np.loadtxt("experimental_data.txt", skiprows=1))
pred = np.array(np.loadtxt("unoscillated_event_rate_prediction.txt", skiprows=1))

x_range = np.linspace(0, 10, 200)
e_range = np.linspace(0.025, 9.975, 200)

def osc_prob(E, theta = (np.pi/4), mass_diff = (2.4), L = 295):
    p = 1 - (np.sin(2*theta)**2)*(np.sin(1.267*mass_diff*1e-3*L/E)**2)
    return p

def gaussian(x, mean, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def convolution(x, spectrum, bin_centers, sigma):
    convolved = 0
    for height, center in zip(spectrum, bin_centers):
        convolved += height*0.05 * gaussian(x, center, sigma)

    return convolved

def nll_1d(bin_number, events, unosc_pred, theta = np.pi/4, mdiff = 2.4, sigma = -24):
    suma = 0
    if sigma == -24:
        for i in range(bin_number):
            osc_pred_i = osc_prob(e_range[i], theta, mdiff)*unosc_pred[i]
            suma += osc_pred_i - events[i]*np.log(osc_pred_i)

    return 2*suma

def nll_2d(bin_number, events, unosc_pred, theta, mdiff):
    suma = 0
    for i in range(bin_number):
        osc_pred = osc_prob(e_range[i], theta, mdiff)*unosc_pred[i]
        suma += osc_pred - events[i]*np.log(osc_pred)
    return 2*suma



def nll_3d_alpha(bin_number, events, unosc_pred, theta = np.pi/4, mdiff = 2.4, alpha = 1.5):
    epsilon = 1e-10
    suma = 0
    for i in range(bin_number):
        osc_pred = osc_prob(e_range[i], theta, mdiff)*unosc_pred[i]*alpha*e_range[i]
        suma += osc_pred - events[i]*np.log(osc_pred + epsilon)
    
    return 2*suma

def nll_4d(bin_number, events, unosc_pred, theta = np.pi/4, mdiff = 2.4, alpha = 0.5, sigma = 0.1):
    suma = 0
    cross_sec_pred = []
    for i in range(bin_number):
        a = osc_prob(e_range[i], theta, mdiff)*unosc_pred[i]
        cross_sec_pred.append(a* alpha* e_range[i])
    for i in range(bin_number):
        convol = convolution(x_range[i], cross_sec_pred, e_range, sigma)
        suma += convol - events[i]*np.log(convol)
    
    return 2*suma


def parabolic_method(func, xmin, xmax, variables, optimiser, params, convergence):
    x = [xmin, (xmin+xmax)/2, xmax]

    counter = 0
    old_x3 = 0

    while True:
        f = []
        for i in x:
            variables[optimiser] = i
            f.append(func(*params, *variables))

        num = (x[2]*x[2]-x[1]*x[1])*f[0] + (x[0]*x[0]-x[2]*x[2])*f[1] + (x[1]*x[1]-x[0]*x[0])*f[2]
        den = (x[2]-x[1])*f[0] + (x[0]-x[2])*f[1] + (x[1]-x[0])*f[2]
        x3 = 1/2 * num/den

        difference = x3 - old_x3
        old_x3 = x3

        x.append(x3)
        variables[optimiser] = x3
        f3 = func(*params, *variables)
        f.append(f3)

        max_index = f.index(max(f))
        del x[max_index]
        del f[max_index]
        
        x.sort()
        
        counter += 1
        
        if np.abs(difference) < convergence:
            break
        
    return x3, f3, counter




#Univariate: parabolic method in both directions and iterate

def univariate_method(func, limits, convergence):
    l = len(limits)

    if type(convergence) is float:
        convergence = np.full(2, convergence)
    convergence = np.array(convergence)

    
    parameters = [200, data, pred]

    traj_x = []
    traj_y = []

    old = []
    new = []
    for i in range(l):
        old.append(0)
        new.append((limits[i][0] + limits[i][1])/2)

    traj_x.append(new[0])
    traj_y.append(new[1])
    counter = 1

    while True:
        #other = []
        k = []
        for i in range(l):
            a, b, d = parabolic_method(func, limits[i][0], limits[i][1], new, i, parameters, convergence[i])
            k.append(a)
            #new[i] = a
            #other.append(b)
        new = k
        #print(new)
        #print(old)
        diff = []
        for i in range(l):
            #limits[i] = [new[i], other[i]]   
            diff.append(np.abs(new[i] - old[i]))
            old[i] = new[i]

        #print(diff)
        #print(convergence)
        counter += 1

        traj_x.append(new[0])
        traj_y.append(new[1])

        c = 0
        for i in range(l):
            if diff[i] < convergence[i]:
                c += 1
        
        if c == l:
            break


    return traj_x, traj_y, counter



#Gradient: taking a step in both dimensions each time

def grad_fwd_2d(func, x, params, h = 1e-7):
    
    f0 = func(*params, *x)
    df_dx = (func(*params, x[0] + h, x[1]) - f0)/h
    df_dy = (func(*params, x[0], x[1] + h) - f0)/h

    return np.array([df_dx, df_dy])

def grad_fwd_3d(func, x, params, h = 1e-7):
    
    f0 = func(*params, *x)
    df_dx = (func(*params, x[0] + h, x[1], x[2]) - f0)/h
    df_dy = (func(*params, x[0], x[1] + h, x[2]) - f0)/h
    df_dz = (func(*params, x[0], x[1], x[2] + h) - f0)/h

    return np.array([df_dx, df_dy, df_dz])

def grad_fwd_4d(func, x, params, h = 1e-7):
    
    f0 = func(*params, *x)
    df_dx = (func(*params, x[0] + h, x[1], x[2], x[3]) - f0)/h
    df_dy = (func(*params, x[0], x[1] + h, x[2], x[3]) - f0)/h
    df_dz = (func(*params, x[0], x[1], x[2] + h, x[3]) - f0)/h
    df_dm = (func(*params, x[0], x[1], x[2], x[3] + h) - f0)/h

    return np.array([df_dx, df_dy, df_dz, df_dm])




def gradient_method(func, initial, convergence, step = 0.0001):
    l = len(initial)
    #scaling = [1e-1, 1e-3]
    #for i in range(l):
    #    convergence[i] /= scaling[i]
    #    initial[i] /= scaling[i]
    
    if type(convergence) is float:
        convergence = np.full_like(initial, convergence)
    parameters = [200, data, pred]

    old = initial

    counter = 0
    
    while True:
        #rescaled = []
        #for j in range(l):
        #    rescaled.append(old[j]*scaling[j])
        grad = np.array([])
        if l == 2:
            grad = grad_fwd_2d(func, old, parameters)
            #print(grad)
        elif l == 3:
            grad = grad_fwd_3d(func, old, parameters)
        elif l == 4:
            grad = grad_fwd_4d(func, old, parameters)

        new = old - step * grad
        #print(new)
        #print(old)
        diff = new - old
        old = new
        counter += 1
        
        c = 0
        for i in range(l):
            if np.abs(diff[i]) < convergence[i]:
                c += 1
        if c == l:
            break

        if counter >= 500:
            break
    

    return new, counter

    


#Newton's: Using the Hessian (second derivatives) with gradient
def hess_2d(func, x, params, h = 1e-7):
   
    f0 = func(*params, *x)
    a = func(*params, x[0] + 2*h, x[1])
    b = func(*params, x[0], x[1] + 2*h)
    c = func(*params, x[0], x[1] + h)
    d = func(*params, x[0] + h, x[1])
    e = func(*params, x[0] + h, x[1] + h)

    h00 = (a - 2*d + f0)/(h**2)
    h11 = (b - 2*c + f0)/(h**2)
    h10 = (e - c - d + f0)/(h**2)
    h01 = h10

    return(np.array([[h00, h01],
                    [h10, h11]]))


def hess_3d(func, x, params, h = 1e-7):
   
    f0 = func(*params, *x)
    a = func(*params, x[0] + 2*h, x[1], x[2])
    b = func(*params, x[0], x[1] + 2*h, x[2])
    c = func(*params, x[0], x[1], x[2] + 2*h)
    d = func(*params, x[0] + h, x[1], x[2])
    e = func(*params, x[0], x[1] + h, x[2])
    f = func(*params, x[0], x[1], x[2] + h)
    g = func(*params, x[0] + h, x[1] + h, x[2])
    i = func(*params, x[0] + h, x[1], x[2] + h)
    j = func(*params, x[0], x[1] + h, x[2] + h)

    h00 = (a - 2*d + f0)/(h**2)
    h11 = (b - 2*e + f0)/(h**2)
    h22 = (c - 2*f + f0)/(h**2)
    h10 = (g - d - e + f0)/(h**2)
    h20 = (i - f - e + f0)/(h**2)
    h21 = (j - d - f + f0)/(h**2)
    h01 = h10
    h02 = h20
    h12 = h21

    return(np.array([[h00, h01, h02],
                    [h10, h11, h12],
                    [h20, h21, h22]]))



def hess_4d(func, x, params, h = 1e-7):
   
    f0 = func(*params, *x)
    a = func(*params, x[0] + 2*h, x[1], x[2], x[3])
    b = func(*params, x[0], x[1] + 2*h, x[2], x[3])
    c = func(*params, x[0], x[1], x[2] + 2*h, x[3])
    d = func(*params, x[0], x[1], x[2], x[3] + 2*h)
    e = func(*params, x[0] + h, x[1], x[2], x[3])
    f = func(*params, x[0], x[1] + h, x[2], x[3])
    g = func(*params, x[0], x[1], x[2] + h, x[3])
    i = func(*params, x[0], x[1], x[2], x[3] + h)
    j = func(*params, x[0] + h, x[1] + h, x[2], x[3])
    k = func(*params, x[0] + h, x[1], x[2] + h, x[3])
    l = func(*params, x[0] + h, x[1], x[2], x[3] + h)
    m = func(*params, x[0], x[1] + h, x[2] + h, x[3])
    n = func(*params, x[0], x[1] + h, x[2], x[3] + h)
    o = func(*params, x[0], x[1], x[2] + h, x[3] + h)

    h00 = (a - 2*e + f0)/(h**2)
    h11 = (b - 2*f + f0)/(h**2)
    h22 = (c - 2*g + f0)/(h**2)
    h33 = (d - 2*i + f0)/(h**2)
    h10 = (j - e - f + f0)/(h**2)
    h20 = (k - e - g + f0)/(h**2)
    h30 = (l - e - i + f0)/(h**2)
    h21 = (m - f - g + f0)/(h**2)
    h31 = (n - f - i + f0)/(h**2)
    h32 = (o - g - i + f0)/(h**2)
    h01 = h10
    h02 = h20
    h12 = h21
    h03 = h30
    h13 = h31
    h23 = h32

    return(np.array([[h00, h01, h02, h03],
                    [h10, h11, h12, h13],
                    [h20, h21, h22, h23],
                    [h30, h31, h32, h33]]))


def newtons_method(func, initial, convergence):
    l = len(initial)
  

    if type(convergence) is float:
        convergence = np.full_like(initial, convergence)
    parameters = [200, data, pred]
    old = initial
    hessian = np.zeros((l, l))
  
    counter = 0

    while True:
        
        grad = np.array([])
        if l == 2:
            grad = grad_fwd_2d(func, old, parameters)
            hessian = hess_2d(func, old, parameters)
        elif l == 3:
            grad = grad_fwd_3d(func, old, parameters)
            hessian = hess_3d(func, old, parameters)
        elif l == 4:
            grad = grad_fwd_4d(func, old, parameters)
            hessian = hess_4d(func, old, parameters)
        

        inv_hess = np.linalg.inv(hessian)
        new = old - np.dot(inv_hess, grad)
        #print(new)
        #print(old)
        #print(np.dot(inv_hess, grad))
        diff = new - old
        #print(diff)
        #print(new)
        #print(old)
        counter += 1
        old = new

        c = 0
        for i in range(l):
            #print(np.abs(diff[i]))
            #print(convergence[i])
            if np.abs(diff[i]) < convergence[i]:
                c += 1

        
        if c == l:
            break
        
        if counter >= 300:
            break

    return new, counter
    #return 1




#Quasi-Newton: approximates the inverse of the hessian

def g_update(g_old, x_update, f_update):
    a = np.outer(x_update, x_update)/np.dot(f_update, x_update)
    b = np.dot(g_old, np.outer(f_update, f_update))
    c = np.dot(f_update, g_old)

    return g_old + a - np.dot(b, g_old)/np.dot(c, f_update)


def quasi_newton_method(func, initial, convergence, step = 0.0001, update_method = "broyden"):
    l = len(initial)
    if type(convergence) is float:
        convergence = np.full_like(initial, convergence)
    parameters = [200, data, pred]
    new = initial
    g = np.identity(l)
    #grad = np.array([])
    #for i in range(len(old)):
    #    np.append(grad, derivative(func, old, i, parameters))

    #new = old - alpha*np.dot(g_initial, grad)

    #grad_new = np.array([])
    #for i in range(len(old)):
    #    np.append(grad, derivative(func, new, i, parameters))
    
    #g_new = g_update(g_initial, new - old, grad_new - grad)
    
    counter = 0

    while True:
        old = new
        grad_new = None
        if l == 2:
            grad = grad_fwd_2d(func, old, parameters)
            new = old - step * np.dot(g, grad)
            grad_new = grad_fwd_2d(func, new, parameters)
            
        elif l == 3:
            grad = grad_fwd_3d(func, old, parameters)
            new = old - step * np.dot(g, grad)
            grad_new = grad_fwd_3d(func, new, parameters)

        elif l == 4:
            grad = grad_fwd_4d(func, old, parameters)
            new = old - step * np.dot(g, grad)
            grad_new = grad_fwd_4d(func, new, parameters)

        grad_diff = grad_new - grad
        diff = new - old
        #outer_diff = np.outer(diff, diff)

        
            
        if update_method == 'broyden':
            r = np.outer(diff - np.dot(g, grad_diff), diff)
            p = np.dot(grad_diff, np.dot(g, grad_diff))
            g = g + np.dot(r, g)/p
        
            
        #g = g_update(g, new - old, grad_new - grad)


        #grad = np.array([])
        #for i in range(len(old)):
            #np.append(grad, derivative(func, old, i, parameters))

        #new = old - alpha*np.dot(g_new, grad)

        #grad_new = np.array([])
        #for i in range(len(old)):
        #    np.append(grad, derivative(func, new, i, parameters))
        
        counter += 1

        c = 0
        for i in range(l):
            #print(np.abs(diff[i]))
            #print(convergence[i])
            if np.abs(diff[i]) < convergence[i]:
                c += 1

        if c == l:
            break
        
        if counter >= 1000:
            break

        
    return new, counter



def error_curv(func, variables, parameters):
    l = len(variables)
    errors = []
    if l == 1:
        h = 1e-8
        f0 = func(*parameters, *variables)
        a = func(*parameters, variables[0] + 2*(h))
        b = func(*parameters, variables[0] + (h))
        d2fdx2 = (a - 2*b + f0)/h**2
        errors = [1/np.sqrt(np.abs(d2fdx2))]

    if l == 2:
        hessian = hess_2d(func, variables, parameters)
        err_x = 1/np.sqrt(np.abs(hessian[0,0]))
        err_y = 1/np.sqrt(np.abs(hessian[1,1]))
        errors = [err_x, err_y]

    if l == 3:
        hessian = hess_3d(func, variables, parameters)
        err_x = 1/np.sqrt(np.abs(hessian[0,0]))
        err_y = 1/np.sqrt(np.abs(hessian[1,1]))
        err_z = 1/np.sqrt(np.abs(hessian[2,2]))
        errors = [err_x, err_y, err_z]
    
    if l == 4:
        hessian = hess_4d(func, variables, parameters)
        err_x = 1/np.sqrt(np.abs(hessian[0,0]))
        err_y = 1/np.sqrt(np.abs(hessian[1,1]))
        err_z = 1/np.sqrt(np.abs(hessian[2,2]))
        err_m = 1/np.sqrt(np.abs(hessian[3,3]))
        errors = [err_x, err_y, err_z, err_m]

    return errors

    


def nll_plusmin_error(theta, parameters):
        plus = nll_1d(*parameters, theta) + 0.5
      
        theta_min_pos = theta
        theta_min_neg = theta
        
        while True:
            theta_min_pos +=0.00001
            
            a = nll_1d(*parameters, theta_min_pos)
            if a >= plus:
                pos_error = theta_min_pos - theta
                break
                 
        while True:
            theta_min_neg -= 0.0001

            b = nll_1d(*parameters, theta_min_neg)
            if b >= plus:
                neg_error = theta_min_neg - theta
                break
        
        return pos_error, neg_error





def test_1d(a, b, c, x):
    """
    1D function with minimum at x = 2
    Used to test parabolic method
    a, b, c are not used, it's just that my minimiser assumes
    the function will have additional parameters
    """
    
    return np.sin(x)**2 + 0.1*x**2


def rosen_2d(a, b, c, x, y):
    """
    2D function with minimum at (x, y) = (1, 1)
    Used to test methods work in 2D
    """

    return (1 - x)**2 + 100 * (y - x**2)**2







