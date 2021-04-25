import sys
import os
# from baseline import run
from baseline3 import run
import csv

run = run
structure = [36, 36]
repetition = 5
fields = ["h1", 'h2', 'left', 'right', 'above', 'below', 'front',
              'behind', 'over', 'under', 'in', 'touch', 'at', 'between',
              'near', 'on top of', 'beside', 'facing', 'total']

n = len(structure)
l = [15,15]
reset_val = 15
jump_val = 2
#l=[1, 1, 1]



def call(i, outputs, write):
    while l[i] <= structure[i]:
        if i < n-1:
            call(i+1, outputs, write)
            l[i] += jump_val
        else:
            for j in range(repetition):
                try:
                    print("Structure: ", l, ", Repetition: ", j)
                    # sys.stdout.flush()
                    # save_stdout = sys.stdout
                    # sys.stdout = open(os.devnull, 'w')
                    rel_acc, acc = run(l)
                    # sys.stdout = save_stdout
                    #outputs += [l + rel_acc + [acc]]
                    write.writerows([l + rel_acc + [acc]])
                    print("rel_acc: ", rel_acc)
                    print("acc: ", acc)
                except Exception as exc:
                    print(exc)
                    continue

            l[i] += jump_val
    l[i] = reset_val

if __name__ == '__main__':

    outputs = []
    f = open('StrctureData.csv', 'a', newline='')
    write = csv.writer(f)
    write.writerow(fields)
    call(0, outputs, write)
    # print(outputs)
    # with open('StrctureData.csv', 'a', newline='') as f:
    #     write = csv.writer(f)
    #     write.writerow(fields)
    #     write.writerows(outputs)