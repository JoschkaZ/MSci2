
import pickle as pkl
import random
import numpy as np

original = pkl.load(open("/home/jz8415/slices2.pkl", 'rb'))



counts = [0,0,0,0,0]
want = int(70000 * 0.25 / 5)

new_data = []
for entry in original:
    img = np.array(entry[0])
    l_z = float(entry[1].split('_z')[1].split('_')[0])

    if counts[int(l_z)-7] < want:

        counts[int(l_z)-7] += 1

        r = (random.random() < 0.0001)
        if r:
            print('o', img.shape)

        img = img[:128,:128]
        if r:
            print('n', img.shape)


        new_data.append([img, entry[1]])


pkl.dump(new_data, open("/home/jz8415/slices2_128.pkl", 'wb'))
