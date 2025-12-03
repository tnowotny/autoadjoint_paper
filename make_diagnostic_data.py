"""
The idea of the dataset made here is to make a spiking dataset with a very long 
time dependency (1000 ms) that is nevetheless fully understandable and use it 
as a diagnostic tool for comparing plain LIF ffwd, recc against LIF with delay/learning
 in ffwd recc versus RAF neurons ffwd, recc ... and with / without delay/learning (?)

Structure of the data is 
t_digit bgnd | t_digit signal | bgnd | t_digit signal | t_digit bgnd

Use Poisson spike trains for signal (f_high) and background (f_low)
(this is guesswork but reflects what SHD/SSC are like?)

Let's do eight input neurons (four pairs), each pair encoding a binary digit of 
classes 0-15.
A bit is set in an input neuron iff neuron 0 spikes in the early window and neuron 1 
in the late window. In all other cases it is zero.
"""

import numpy as np
import matplotlib.pyplot as plt
import mnist


def fill_Poisson_events(t0, t1, d, x, p, r):
    """
    fill a list d with tuples (t,x,p) where t0 < t < t1 are Poisson spike times
    with rate r
    """
    t = t0
    done = False
    while not done:
        ts = np.random.exponential(scale=1./r)
        if t+ts < t1:
            t = t+ts
            d.append((t*1000.0,x,p))
        else:
            done = True
    return d

def fill_Poisson_list(t0, t1, r):
    """
    fill a list d with times t where t0 < t < t1 are Poisson spike times
    with rate r
    """
    t = t0
    done = False
    d = []
    while not done:
        ts = np.random.exponential(scale=1./r)
        if t+ts < t1:
            t = t+ts
            d.append(t)
        else:
            done = True
    return d

""" 
This function makes diagnostic data that consists of digits that are encoded as
bits. Each bit is represented as a high firing rate in one neuron at the front of the
trial and in a second neuron at the end of the trial. So there are 2*(number of bits) 
input neurons. 
""" 
def generate_diag_digit_data(T, t_digit, l_low, l_high, num_per_class, bits, plot=False):
    mytype = np.dtype([('t',np.float32),('x',np.int8),('p',np.int8)])
    data = []
    label = []
    p = 0
    for c in range(2**bits):
        bit_array = [(c >> i) & 1 for i in range(bits)][::-1]
        for k in range(num_per_class):
            d = []
            print(bit_array)
            for x,b in enumerate(bit_array):
                lbd = l_high if b == 1 else l_low
                d = fill_Poisson_events(0,t_digit,d,2*x,p,l_low)
                d = fill_Poisson_events(t_digit,2*t_digit,d,2*x,p,lbd)
                d = fill_Poisson_events(2*t_digit,T,d,2*x,p,l_low)
                
                d = fill_Poisson_events(0,T-2*t_digit,d,2*x+1,p,l_low)
                d = fill_Poisson_events(T-2*t_digit,T-t_digit,d,2*x+1,p,lbd)
                d = fill_Poisson_events(T-t_digit,T,d,2*x+1,p,l_low)
            data.append((np.array(d,dtype=mytype),c))
            if (plot):
                plt.figure()
                plt.scatter(data[c*num_per_class+k]["t"],data[c*num_per_class+k]["x"],s=1)
                plt.title(f"class {c}")
                plt.show()
            
    return data


"""
This functions creates an XOR task where there is high firing for 1 and baseline 
firing for 0. There are two input neurons, one has the encoded bit int he front, the other in the back. The two output classes are determined by the value of XOR of the fron and the back bit. Reverts to the normal XOR between two neurons if teh length of the trial is such that the back window coincides with the front window.
"""

def generate_diag_xor_data(T, t_digit, l_low, l_high, num_per_class, plot=False):
    mytype = np.dtype([('t',np.float32),('x',np.int8),('p',np.int8)])
    data = []
    label = []
    p = 0
    r = [ l_low, l_high ]
    for front in range(2):
        for back in range(2):
            c = 1 if front*back == 0 and front+back > 0 else 0
            for k in range(num_per_class):
                d = []
                d = fill_Poisson_events(0,t_digit,d,0,p,l_low)
                d = fill_Poisson_events(t_digit,2*t_digit,d,0,p,r[front])
                d = fill_Poisson_events(2*t_digit,T,d,0,p,l_low)
                
                d = fill_Poisson_events(0,T-2*t_digit,d,1,p,l_low)
                d = fill_Poisson_events(T-2*t_digit,T-t_digit,d,1,p,r[back])
                d = fill_Poisson_events(T-t_digit,T,d,1,p,l_low)
                data.append((np.array(d,dtype=mytype),c))
                if (plot):
                    plt.figure()
                    plt.scatter(data[-1][0]["t"],data[-1][0]["x"],s=1)
                    plt.title(f"class {c}")
                    plt.show()
            
    return data

"""
Different encoding of XOR that is more amenable to SNN processing; 
There are 4 input neurons, two each for a binary variable: neuron 0 on means 0 and 
neuron1 on means 1. 
"""

def generate_xor_data_identity_coding_Poisson(T, t_digit, l_low, l_high, num_per_class, plot=False):
    t = []
    ids = []
    label = []
    p = 0
    r = [ l_low, l_high ]
    for front in range(2):
        for back in range(2):
            c = 1 if front*back == 0 and front+back > 0 else 0
            for k in range(num_per_class):
                lt = []
                lid = []
                for n in range(4):
                    if n == 0:
                        the_r = r[(front+1)%2]
                        t_signal = t_digit
                    elif n == 1:
                        the_r = r[front]
                        t_signal = t_digit
                    elif n == 2:
                        the_r = r[(back+1)%2]
                        t_signal = T-2*t_digit
                    elif n == 3:
                        the_r = r[back]
                        t_signal = T-2*t_digit
                    t1 = fill_Poisson_list(0,t_signal,l_low)
                    t2 = fill_Poisson_list(t_signal,t_signal+t_digit,the_r)
                    t3 = fill_Poisson_list(t_signal+t_digit,T,l_low)
                    lt += t1+t2+t3
                    lid += [n]*(len(t1)+len(t2)+len(t3))
                t.append(lt)
                ids.append(lid)
                label.append(c)
                if (plot):
                    plt.figure()
                    plt.scatter(lt,lid,s=1)
                    plt.title(f"class {c}")
                    plt.show()
            
    return t, ids, label

def pick_spike_times(t0, t1, n, t_refr):
    fac = 1.0/t_refr
    lt = np.ones(n)*2*t_refr+t1
    rng = np.random.default_rng()
    for i in range(n):
        while True:
            x = rng.uniform(t0, t1)
            #print(x)
            if np.min(np.abs(lt-x)) > t_refr:
                break
        lt[i] = x
    return list(lt)

        
def generate_xor_data_identity_coding(T, t_digit, n_spike, t_refr, r_noise, num_per_class, plot=False):
    t = []
    ids = []
    label = []
    p = 0
    for front in range(2):
        for back in range(2):
            c = 1 if front*back == 0 and front+back > 0 else 0
            for k in range(num_per_class):
                lt = []
                lid = []
                for n in range(4):
                    if n == 0:
                        on = front == 0
                        t_signal = t_digit
                    elif n == 1:
                        on = front == 1
                        t_signal = t_digit
                    elif n == 2:
                        on = back == 0
                        t_signal = T-2*t_digit
                    elif n == 3:
                        on = back == 1
                        t_signal = T-2*t_digit
                    if on:
                        t1 = fill_Poisson_list(0,t_signal,r_noise)
                        t2 = pick_spike_times(t_signal, t_signal+t_digit, n_spike, t_refr)
                        t3 = fill_Poisson_list(t_signal+t_digit,T,r_noise)
                        lt += t1+t2+t3
                        lid += [n]*(len(t1)+len(t2)+len(t3))
                    else:
                        t1 = fill_Poisson_list(0,T,r_noise)
                        lt += t1
                        lid += [n]*len(t1)
                        
                t.append(lt)
                ids.append(lid)
                label.append(c)
                if (plot):
                    plt.figure()
                    plt.scatter(lt,lid,s=1)
                    plt.title(f"class {c}")
                    plt.xlim([ 0, T])
                    plt.show()
            
    return t, ids, label

"""
This function creates an MNIST-based delayed adding task. Two MNIST digits are 
latency encoded and spikes are generated in two populations of 28x28 neurons. In population 1, spikes of digit 1 are generated in the front of the trial. In population 2, spikes of digit 2 are generated after a delay/ at the end of the trial. The class is the sum of the two digits.
"""

"""
This function loads MNIST data and makes a train/validation split on the shuffled training set to proportions
that make sense with the later goal sizes for the training and validation sets of paired images.
N_train: The desired later number of training examples of paired images
N_val: The desired later number of vlidation examples of paired images
"""

def load_MNIST_data(N_train, N_val):
    mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    labels_train = mnist.train_labels()
    images_train = mnist.train_images()
    labels_test = mnist.test_labels()
    images_test = mnist.test_images()
    # shuffle training set
    idx = np.arange(len(labels_train))
    rng = np.random.default_rng()
    rng.shuffle(idx)
    labels_train = labels_train[idx]
    images_train = images_train[idx]
    # split
    n_train = int(np.sqrt(N_train)/(np.sqrt(N_train)+np.sqrt(N_val))*len(labels_train))
    n_val = len(labels_train)- n_train
    images_val = images_train[n_train:]
    labels_val = labels_train[n_train:]
    images_train = images_train[:n_train]
    labels_train = labels_train[:n_train]
    return labels_train, images_train, labels_val, images_val, labels_test, images_test

"""
N_ex: desired number of examples
images: single MNIST images to use
labels: matching MNIST labels
delay: time between the two latency encoded spike trains
min_time: front time buffer for each individual latency encoding
max_time: time for the last spikes in teh encoding
thresh: threshold value for the pixel colour to generate a spike
plot: whether to make an example diagnostic plot
"""

def generate_latency_MNIST_sum_examples(N_ex, images, labels, delay, min_time= 0.0, max_time= 30.0, thresh= 1, r_noise= 0.0, plot= False):
    # make the paired data
    t = []
    ids = []
    lab= []
    time_range = max_time - min_time
    rng = np.random.default_rng()
    N_s = len(labels)
    for n in range(N_ex):
        i = rng.integers(0, N_s, 2)
        spike_vector = images[i[0]] > thresh
        # Extract values of spiking pixels
        spike_pixels = images[i[0], spike_vector]
        # Calculate spike times
        times = (((255.0 - spike_pixels) / 255.0) * time_range) + min_time
        sids = np.where(spike_vector.flatten())[0]
            
        spike_vector = images[i[1]] > thresh
        # Extract values of spiking pixels
        spike_pixels = images[i[1], spike_vector]
        # Calculate spike times
        times2= (((255.0 - spike_pixels) / 255.0) * time_range) + min_time + delay
        sids2 = np.where(spike_vector.flatten())[0]+784
        noise = []
        noise_id = []
        if r_noise > 0.0:
            #for id in range(2*784):
            #    noise.append(fill_Poisson_list(0.0,delay+max_time, r_noise))
            #    noise_id.append([id]*len(noise[-1]))
            rng = np.random.default_rng()
            n_noise = int(2*784*r_noise*(delay+max_time))
            noise= rng.uniform(0.0, delay+max_time, n_noise)
            noise_id= rng.integers(0, 2*784, n_noise)
        t.append(np.hstack([times,times2,noise]))
        the_id = np.hstack([list(sids),list(sids2),noise_id])
        ids.append(the_id.astype(int))
        lab.append(labels[i[0]]+labels[i[1]])
        #print(t[-1])
        #print(ids[-1])
        #print(lab[-1])
    
        if plot:
            plt.figure()
            plt.imshow(images[i[0]])
            plt.figure()
            plt.imshow(images[i[1]])
            plt.figure()
            plt.scatter(t[-1],ids[-1])
            plt.show()
    return t, ids, lab
    
def generate_latency_MNIST_sum(N_train, N_val, N_test, delay, min_time= 0.0, max_time= 30.0, thresh= 1, plot= False):
    mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    labels_train = mnist.train_labels()
    images_train = mnist.train_images()
    labels_test = mnist.test_labels()
    images_test = mnist.test_images()
    n_test = len(labels_test)
    
    # shuffle training set
    idx = np.arange(len(labels_train))
    rng = np.random.default_rng()
    rng.shuffle(idx)
    labels_train = labels_train[idx]
    images_train = images_train[idx]

    # make validation split on single digits
    n_train = int(np.sqrt(N_train)/(np.sqrt(N_train)+np.sqrt(N_val))*len(labels_train))
    n_val = len(labels_train)- n_train
    images_val = images_train[n_train:]
    labels_val = labels_train[n_train:]
    images_train = images_train[:n_train]
    labels_train = labels_train[:n_train]

    # make the paired data
    t_train = []
    id_train = []
    lab_train = []
    t_val = []
    id_val = []
    lab_val= []
    t_test = []
    id_test = []
    lab_test = []
    time_range = max_time - min_time
    for images, labels, N_s, t, ids, lab, N in [(images_train, labels_train, n_train, t_train, id_train, lab_train, N_train),
                                     (images_val, labels_val, n_val, t_val, id_val, lab_val, N_val),
                                     (images_test, labels_test, n_test, t_test, id_test, lab_test, N_test)]:
        for n in range(N):
            i = rng.integers(0, N_s, 2)
            spike_vector = images[i[0]] > thresh
            # Extract values of spiking pixels
            spike_pixels = images[i[0], spike_vector]
            # Calculate spike times
            times = (((255.0 - spike_pixels) / 255.0) * time_range) + min_time
            sids = np.where(spike_vector.flatten())[0]
            
            spike_vector = images[i[1]] > thresh
            # Extract values of spiking pixels
            spike_pixels = images[i[1], spike_vector]
            # Calculate spike times
            times2= (((255.0 - spike_pixels) / 255.0) * time_range) + min_time + delay
            sids2 = np.where(spike_vector.flatten())[0]+784
            t.append(np.hstack([times,times2]))
            ids.append(np.hstack([sids,sids2]))
            lab.append(labels[i[0]]+labels[i[1]])
            #print(t[-1])
            #print(ids[-1])
            #print(lab[-1])
    
            if plot:
                plt.figure()
                plt.imshow(images[i[0]])
                plt.figure()
                plt.imshow(images[i[1]])
                plt.figure()
                plt.scatter(t[-1],ids[-1])
                plt.show()
    return t_train, id_train, lab_train, t_val, id_val, lab_val, t_test, id_test, lab_test


def generate_timing_xor(t_0, t_1, t_delay, jitter_std, N_samples, r_noise= 0.0, plot=False):
    ts = [ t_0, t_1 ]
    all_t = []
    all_id = []
    label = []
    for i in range(N_samples):
        for front in range(2):
            for back in range(2):
                c = 1 if front*back == 0 and front+back > 0 else 0
                t = [ ts[front]+np.random.normal(0,jitter_std), t_delay+ts[back]+np.random.normal(0,jitter_std) ]
                id = [ 0, 0 ]
                if r_noise > 0.0:
                    ns_t = fill_Poisson_list(0, t_1+t_delay, r_noise)
                    ns_id = [1]*len(ns_t)
                    t= t+ns_t
                    id= id + ns_id
                all_t.append(t)
                all_id.append(id)
                label.append(c)
            
    return all_t, all_id, label

def generate_Poisson_only(t_total, r_noise, N_samples):
    all_t = []
    all_id = []    
    for i in range(N_samples):
        ns_t = fill_Poisson_list(0, t_total, r_noise)
        ns_id = [0]*len(ns_t)
        all_t.append(ns_t)
        all_id.append(ns_id)
    return all_t, all_id


if __name__ == "__main__":
    #generate_diag_digit_data(1000, 100, 0.01, 0.1, 1, 4, plot=True)
    #generate_diag_xor_data(300, 100, 0.02, 0.2, 5, plot=True)
    #generate_latency_MNIST_sum(200000, 20000, 100000, 500.0, plot= False)
    #generate_xor_data_identity_coding_Poisson(60.0, 20.0, 0.1, 1.0, 2, plot=True)
    #print(pick_spike_times(10, 30, 5, 3))
    generate_xor_data_identity_coding(150.0, 50.0, 10, 1.0, 0.02, 5, plot=True)
