
import pickle as pkl
import random


original = pkl.load(open("/home/jz8415/slices2.pkl", 'rb'))

print(original.shape)

counts = [0,0,0,0,0]
want = int(70000 * 0.25 / 5)

new_data = []
for entry in original:
    img = entry[0]
    label = int(entry[1])

    if counts[label-7] < want:

        counts[label-7] += 1

        r = (random.random() < 0.0001)
        if r:
            print('o', img.shape)

        img = img[:128,:128]
        if r:
            print('n', img.shape)


        new_data.append([img, int(label)])

print(new_data.shape)
pkl.dump(new_data, open("/home/jz8415/slices2_128.pkl"), 'wb')
